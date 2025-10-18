import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from model import SRCNN
import os
import sys

def upscale_image(input_path, output_path, scale=2, use_cpu=False):
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = SRCNN()
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()
    model.to(device)

    image = Image.open(input_path).convert('RGB')
    w, h = image.size
    
    new_w = w * scale
    new_h = h * scale

    image = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BICUBIC)

    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output = output.squeeze(0).cpu()
    to_pil = transforms.ToPILImage()
    output_image = to_pil(output.clamp(0, 1))

    output_image.save(output_path)
    print(f'Upscaled image saved to {output_path}')

if __name__ == '__main__':
    scale = 2
    use_cpu = False
    
    all_scales = False
    if '--cpu' in sys.argv:
        use_cpu = True
        sys.argv.remove('--cpu')
    if '--all' in sys.argv:
        all_scales = True
        sys.argv.remove('--all')

    if len(sys.argv) > 1:
        try:
            scale = int(sys.argv[1])
        except ValueError:
            print('Invalid scale argument, defaulting to 2')

    input_dir = 'input'
    output_dir = 'output'

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print(f'No images found in {input_dir} folder')
    else:
        for image_file in image_files:
            input_path = os.path.join(input_dir, image_file)
            name_without_ext, ext = os.path.splitext(image_file)
            if all_scales:
                for s in [2, 4, 6, 8, 10]:
                    output_filename = f'{name_without_ext}_{s}x{ext}'
                    output_path = os.path.join(output_dir, output_filename)
                    print(f'Processing {image_file} with {s}x upscaling...')
                    upscale_image(input_path, output_path, scale=s, use_cpu=use_cpu)
            else:
                output_filename = f'{name_without_ext}_{scale}x{ext}'
                output_path = os.path.join(output_dir, output_filename)
                print(f'Processing {image_file} with {scale}x upscaling...')
                upscale_image(input_path, output_path, scale=scale, use_cpu=use_cpu)