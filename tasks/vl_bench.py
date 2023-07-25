import json
from tqdm import tqdm
import logging

import torch
import torch.backends.cudnn as cudnn

from models.model_retrieval_base import SingularityRetrievalBase
from utils.config_utils import setup_config, setup_evaluate_config
from utils.basic_utils import setup_seed
from dataset.vl_bench import setup_loader, process_path
from models.tokenization_bert import BertTokenizer
from models.utils import (
    interpolate_pos_embed,
    interpolate_pos_relative_bias_beit,
    load_temp_embed_with_mismatch,
)
import copy

logger = logging.getLogger(__name__)



class CustomModel(SingularityRetrievalBase):
    def __init__(self, config=None, tokenizer=None):
        super(CustomModel, self).__init__(
            config=config, tokenizer=tokenizer, pretrain=False
        )
    
    def forward(self, image, text):
        # ================= Dual Encoder ITC loss ================ #
        self.clip_contrastive_temperature()

        image_embeds, pooled_image_embeds = self.encode_image(image)
        text_embeds, pooled_text_embeds = self.encode_text(text)

        sim_i2t, sim_t2i = self.get_sim(
            pooled_image_embeds, pooled_text_embeds, t=self.temp)
        return sim_i2t
    

def setup_model(config, has_decoder=False, pretrain=False, find_unused_parameters=False):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    tokenizer = BertTokenizer.from_pretrained(config.text_encoder)
    model = CustomModel(config=config, tokenizer=tokenizer)

    model = model.to(torch.device(config.device))
    if not config.pretrained_path:
        raise KeyError("This needs a pretrained model.")

    logger.info(f"Loading checkpoint from {config.pretrained_path}")
    checkpoint = torch.load(config.pretrained_path, map_location="cpu")
    state_dict = checkpoint["model"]

    # reshape positional embeddings
    is_beit = "beit" in config.vit_type
    if is_beit:
        # interpolate relative pos bias
        state_dict = interpolate_pos_relative_bias_beit(
            state_dict_old=state_dict,
            state_dict_new=model.state_dict(),
            patch_shape_new=model.vision_encoder.embeddings.patch_embeddings.patch_shape
        )
    else:
        # interpolate pos_embed
        state_dict["vision_encoder.embeddings.position_embeddings"] = \
            interpolate_pos_embed(
                pos_embed_old=state_dict["vision_encoder.embeddings.position_embeddings"],
                pos_embed_new=model.vision_encoder.embeddings.position_embeddings,
                num_patches_new=model.vision_encoder.embeddings.patch_embeddings.num_patches
            )

    # load temporal_embeddings, clip or expand when necessary
    state_dict["temporal_embeddings"] = load_temp_embed_with_mismatch(
        temp_embed_old=state_dict["temporal_embeddings"],
        temp_embed_new=model.temporal_embeddings.data
    )

    for key in list(state_dict.keys()):
        if "bert" in key:
            encoder_key = key.replace("bert.", "")
            state_dict[encoder_key] = state_dict[key]
            if not has_decoder:
                del state_dict[key]

        # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
        # only for generation tasks like VQA
        if has_decoder and "text_encoder" in key:
            if "layer" in key:
                encoder_keys = key.split(".")
                layer_num = int(encoder_keys[4])
                if layer_num < 9:  # configs/config_bert.fusion_layer
                    del state_dict[key]
                    continue
                else:
                    decoder_layer_num = (layer_num-9)
                    encoder_keys[4] = str(decoder_layer_num)
                    encoder_key = ".".join(encoder_keys)
            else:
                encoder_key = key
            decoder_key = encoder_key.replace("text_encoder", "text_decoder")
            state_dict[decoder_key] = state_dict[key]
            del state_dict[key]

    # load temporal_embeddings, clip or expand when necessary
    state_dict["temporal_embeddings"] = load_temp_embed_with_mismatch(
        temp_embed_old=state_dict["temporal_embeddings"],
        temp_embed_new=model.temporal_embeddings.data
    )

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(msg)
    logger.info(f"Loaded checkpoint from {config.pretrained_path}")

    if config.fp16:
        model = model.half()
    return model, tokenizer


@torch.no_grad()
def main(config):
    logger.info(f"config: \n{config}")
    setup_seed(config.seed)
    device = torch.device(config.device)
    cudnn.benchmark = True

    # TODO: hack the following line
    # train_loaders, test_name2loaders, train_media_types = setup_dataloaders(config, mode="ret")

    loader = setup_loader(config)
    model, tokenizer = setup_model(
        config,
        has_decoder=False,
        pretrain=False,
        find_unused_parameters=True
    )

    results = dict()
    for i, batch in enumerate(tqdm(loader)):
        video = batch['video'].to(device)
        text_input = tokenizer(
            batch['texts'],
            padding="max_length",
            truncation=True,
            max_length=config.max_txt_l,
            return_tensors='pt',
        ).to(device)
        with torch.cuda.amp.autocast(enabled=config.fp16):
            sim_i2t = model(video, text_input)  # size: V, T
            sim_i2t = sim_i2t.to('cpu')
        processed = post_process_sim(
            sim_i2t,
            batch['ids'],
            batch['num_texts'],
        )
        for item_id, scores in processed:
           results[item_id] = {'scores': scores}

    with open(process_path(config.output_file), 'w') as f:
        json.dump(results, f, sort_keys=False, indent=4)


def post_process_sim(sim_i2t, item_ids, num_texts):
    num_videos = sim_i2t.shape[0]
    num_total_texts = sum(num_texts)
    assert num_total_texts == sim_i2t.shape[1]
    offset = 0
    results = list()
    for i in range(num_videos):
        scores = sim_i2t[i, offset:offset+num_texts[i]].tolist()
        results.append((item_ids[i], scores))
        offset += num_texts[i]
    return results


if __name__ == "__main__":
    config = setup_config()
    main(config)
