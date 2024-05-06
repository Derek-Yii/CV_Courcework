from ultralytics import YOLO
import argparse
import warnings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='snake')
    opt = parser.parse_args()
    model = YOLO(f'./runs/train/{opt.model}/weights/best.pt')   
    model.predict(
        source=r'./video',
        save=True, 
        imgsz=640,  
        conf=0.3,  
        iou=0.45,  
        show=False, 
        project='runs', 
        name='predict_person',  
        save_txt=False,  
        save_conf=True,  
        save_crop=False,  
        show_labels=True,  
        show_conf=False,  
        vid_stride=1, 
        line_width=2,  
        visualize=False, 
        augment=False, 
        agnostic_nms=False,  
        retina_masks=False,  
        boxes=True,  
        
    )

