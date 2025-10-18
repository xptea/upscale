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
    output_image = to_pil(output.clamp(0, 1))  # Clamp to [0,1]

    output_image.save(output_path)
    print(f'Upscaled image saved to {output_path}')

if __name__ == '__main__':
    scales = [2, 4, 6, 8, 10]
    use_cpu = False
    
    if '--cpu' in sys.argv:
        use_cpu = True
        sys.argv.remove('--cpu')
    
    input_dir = 'input'
    output_dir = 'output'
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f'No images found in {input_dir} folder')
    else:
        for image_file in image_files:
            input_path = os.path.join(input_dir, image_file)
            name_without_ext = os.path.splitext(image_file)[0]
            ext = os.path.splitext(image_file)[1]
            
            for scale in scales:
                output_filename = f'upscaled_{name_without_ext}_{scale}x{ext}'
                output_path = os.path.join(output_dir, output_filename)
                print(f'Processing {image_file} with {scale}x upscaling...')
                upscale_image(input_path, output_path, scale=scale, use_cpu=use_cpu)