import torch
import torch.nn as nn
from model.decoder_aster import Decoder
from model.encoder_aster import Encoder
from model.loss import SequenceCrossEntropyLoss


class MathsNet(nn.Module):
    """
     This is the integrated model.
     """

    def __init__(self, config, training, device, use_beam=False, beam_width=5):
        super(MathsNet, self).__init__()
        self.rec_num_classes = config['num_classes']
        self.eos_symbol_id = config['end_id']
        self.training = training
        self.use_beam = use_beam
        self.beam_width = beam_width
        self.device = device

        # modules
        self.encoder = Encoder()
        self.decoder = Decoder(
            num_classes=self.rec_num_classes,
            in_planes=self.encoder.out_planes,
            device=device,
        )
        self.rec_crit = SequenceCrossEntropyLoss(eos_symbol_id=self.eos_symbol_id)

        # load weights
        if training:
            pass
            state_dict = torch.load(config['pretrain'])['state_dict']
            # filter unusable weights
            pretrained_dict = {
                k: v
                for k, v in state_dict.items()
                if (k in self.state_dict() and 'encoder.layer0' not in k
                    and 'decoder.fc' not in k
                    and 'decoder.tgt_embedding' not in k)
            }
            self.load_state_dict(
                pretrained_dict,
                strict=False,
            )
        else:
            self.load_state_dict(torch.load(config['checkpoint']), strict=True)

    def forward(self, input_dict):
        return_dict = {}
        return_dict['output'] = {}

        x, rec_targets, rec_lengths = (
            input_dict['img'],
            input_dict['label'],
            len(input_dict['label'][0])
        )

        encoder_feats = self.encoder(x)
        encoder_feats = encoder_feats.contiguous()

        if self.use_beam:
            return_dict['output']['pred_rec'], _ = self.decoder.beam_search(encoder_feats, beam_width=self.beam_width)
        else:
            return_dict['output']['pred_rec'], _ = self.decoder.predict(encoder_feats)

        rec_pred = self.decoder([encoder_feats, rec_targets, rec_lengths])
        return_dict['loss'] = self.rec_crit(rec_pred, rec_targets, rec_lengths)
        return_dict['loss'] = return_dict['loss'].unsqueeze(0)

        return return_dict
