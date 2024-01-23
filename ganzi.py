from models import pix2pix_model, networks
from torchvision import transforms
import base64, cv2
import numpy as np
import torch
import argparse
from PIL import Image
import io

class MyModel:
    device = "cpu"
    netG_dict = dict()
    w, h = 720, 480
    transform = transforms.Compose(
        [transforms.CenterCrop((h // 2, w // 2)), transforms.Resize((512, 512)), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        # arcane model opt
        arcane_opt = argparse.Namespace(
            input_nc=3,
            output_nc=3,
            ngf=64,
            netG='unet_256',
            norm='batch',  # norm='instance'
            use_dropout=False,
            init_type='normal',
            init_gain=0.02,
            gpu_ids=[],
            activation='swish',
            squeeze=4
        )

        # pixar model opt
        pixar_opt = argparse.Namespace(
            input_nc=3,
            output_nc=3,
            ngf=64,
            netG='unet_256',
            norm='batch',
            use_dropout=False,
            init_type='normal',
            init_gain=0.02,
            gpu_ids=[],
            activation='swish',
            squeeze=4
        )

        # disney model opt
        disney_opt = argparse.Namespace(
            input_nc=3,
            output_nc=3,
            ngf=64,
            netG='unet_256',
            norm='batch',  # norm='instance'
            use_dropout=False,
            init_type='normal',
            init_gain=0.02,
            gpu_ids=[],
            activation='swish',
            squeeze=4
        )

        arcane_netG = networks.define_G(arcane_opt.input_nc, arcane_opt.output_nc, arcane_opt.ngf, arcane_opt.netG, arcane_opt.norm, arcane_opt.use_dropout,
                                 arcane_opt.init_type, arcane_opt.init_gain, arcane_opt.gpu_ids, arcane_opt.activation, arcane_opt.squeeze)

        pixar_netG = networks.define_G(pixar_opt.input_nc, pixar_opt.output_nc, pixar_opt.ngf, pixar_opt.netG, pixar_opt.norm, pixar_opt.use_dropout,
                                 pixar_opt.init_type, pixar_opt.init_gain, pixar_opt.gpu_ids, pixar_opt.activation, pixar_opt.squeeze)

        disney_netG = networks.define_G(disney_opt.input_nc, disney_opt.output_nc, disney_opt.ngf, disney_opt.netG, disney_opt.norm, disney_opt.use_dropout,
                                 disney_opt.init_type, disney_opt.init_gain, disney_opt.gpu_ids, disney_opt.activation, disney_opt.squeeze)

        arcane_path = "./ML/arcane/latest_net_G.pth"
        pixar_path = "./ML/pixar/latest_net_G.pth"
        disney_path = "./ML/disney/latest_net_G.pth"

        order_dict_arcane = torch.load(arcane_path)
        order_dict_pixar = torch.load(pixar_path)
        order_dict_disney = torch.load(disney_path)

        arcane_netG.load_state_dict(order_dict_arcane)
        pixar_netG.load_state_dict(order_dict_pixar)
        disney_netG.load_state_dict(order_dict_disney)

        arcane_netG.eval()
        pixar_netG.eval()
        disney_netG.eval()

        self.netG_dict["arcane"] = arcane_netG
        self.netG_dict["pixar"] = pixar_netG
        self.netG_dict["disney"] = disney_netG
        pass

    # 이미지 변환 함수 input_data = 클라이언트가 보낸 jpeg , model = 어떤 모델을 쓸건지 string 으로 받아옴
    async def image_handler(self, input_data, model):
        # 모델 예측 로직
        # print(f'model = {model}, data = {input_data}')
        image_data = base64.b64decode(input_data)
        image = Image.open(io.BytesIO(image_data))
        t_img = self.transform(image)
        t_img = t_img.view(1, 3, 512, 512)
        t_img = t_img.to(self.device)
        #print(model)
        netG = self.netG_dict.get(model)
        netG = netG.to(self.device)  # 이부분에서 성능 ?
        with torch.no_grad():
            out = netG(t_img)
        out = out.to('cpu')
        out = out * 0.5 + 0.5
        image_np = out[0].numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * 255).astype(np.uint8)
        opencv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(opencv_image, (640, 360))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
        processed_img_data = base64.b64encode(frame_encoded).decode()
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + processed_img_data
        return processed_img_data
        pass