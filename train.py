import argparse
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--models', default='yolov8.yaml', help='model.yaml path')
    parser.add_argument('--name', default='yolov8', help ='the folder of the seved model'  )
    parser.add_argument('--data', default='data.yaml')
    opt = parser.parse_args()
    model = YOLO(f'ultralytics/cfg/models/v8/{opt.models}')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=opt.data,
                cache=False,
                imgsz=640,
                epochs=opt.epochs,
                batch=opt.batch,
                close_mosaic=10,
                workers=8,
                device=opt.device,
                optimizer='SGD', # using SGD
                project='runs/train',
                name=opt.name,
                patience=50,
                lr0=0.01, 
                lrf=0.009,  
                momentum=0.937,  
                weight_decay=0.0005,  
                warmup_epochs=3.0,  
                warmup_momentum=0.8,  
                warmup_bias_lr=0.1,  
                box=7.5,  
                cls=0.5, 
                dfl=1.5,  
                pose=12.0,  
                kobj=1.0,  
 	        label_smoothing=0.0,  
                nbs=64,  
                hsv_h=0.015,  
                hsv_s=0.7,  
                hsv_v=0.55,  
                degrees=0.0,  
                translate=0.2,  
                scale=0.6,  
                shear=0.0,  
                perspective=0.0,  
                flipud=0.0, 
                fliplr=0.6,  
                mosaic=1.0, 
                mixup=0.5, 
                 
                
                )
