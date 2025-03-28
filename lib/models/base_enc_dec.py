import torch
import torch.nn as nn
from functools import partial


__all__ = ["EncoderDecoder", 
           "E_vit_D_xxx", 
           "Encoder1Decoder12", 
           "E1_vit_D12_xxx",
           "Encoder1Decoder123"
           ]

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super().__init__()
        if encoder is None:
            self.encoder = nn.Identity()
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = nn.Identity()
        else:
            self.decoder = decoder

        if kwargs.get('encoder_forward'):
            attrname = kwargs.get('encoder_forward')
            setattr(self.encoder, 'forward', getattr(self.encoder, attrname))

        if kwargs.get('encoder_preprocess'):
            self.encoder_preprocess = kwargs.get('encoder_preprocess')
        else:
            self.encoder_preprocess = None
        if self.encoder_preprocess is None:
            self.encoder_preprocess = lambda x: x

        if kwargs.get('decoder_preprocess'):
            self.decoder_preprocess = kwargs.get('decoder_preprocess')
        else:
            self.decoder_preprocess = None
        if self.decoder_preprocess is None:
            self.decoder_preprocess = lambda x: x

        if kwargs.get('decoder_postprocess'):
            self.decoder_postprocess = kwargs.get('decoder_postprocess')
        else:
            self.decoder_postprocess = None
        if self.decoder_postprocess is None:
            self.decoder_postprocess = lambda x: x

    def forward(self, x):
        x = self.encoder_preprocess(x)
        x = self.encoder(x)

        x = self.decoder_preprocess(x)
        x = self.decoder(x)
        x = self.decoder_postprocess(x)
        return x

def E_vit_D_xxx(encoder, decoder, encoder_forward="forward_features"):
    return EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        encoder_forward=encoder_forward,
        decoder_preprocess=lambda x: x[:, 0],
    )

class Encoder1Decoder12(EncoderDecoder):
    def forward_(self, x):
        x1, x2 = x
        return self.backup(x1, x2)
    
    def backup(self, x1, x2):
        x1 = self.encoder_preprocess(x1)
        x1 = self.encoder(x1)

        x1 = self.decoder_preprocess(x1)
        x = self.decoder(x1, x2)
        x = self.decoder_postprocess(x)

        return x
    
    def forward(self, x1, x2):
        x1 = self.encoder_preprocess(x1)
        x1 = self.encoder(x1)

        x1 = self.decoder_preprocess(x1)
        x = self.decoder(x1, x2)
        x = self.decoder_postprocess(x)

        return x
    
class Encoder1Decoder123(EncoderDecoder):
    def forward(self, x1, x2, x3):
        x1 = self.encoder_preprocess(x1)
        x1 = self.encoder(x1)

        x1 = self.decoder_preprocess(x1)
        x = self.decoder(x1, x2, x3)
        x = self.decoder_postprocess(x)

        return x
    
def E1_vit_D12_xxx(encoder, decoder, encoder_forward="forward_features"):
    return Encoder1Decoder12(
        encoder=encoder,
        decoder=decoder,
        encoder_forward=encoder_forward,
        decoder_preprocess=lambda x: x[:, 0],
    )

