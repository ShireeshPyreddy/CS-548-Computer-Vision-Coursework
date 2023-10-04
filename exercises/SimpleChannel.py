from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D
import cv2
import numpy as np

def main():
    input_node = Input(shape=(None, None, 3))
    filter_layer = Dense(1, use_bias=False)
    output_node = filter_layer(input_node)
    model = Model(inputs=input_node, outputs=output_node)
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print("ERROR: Cannot open camera!")
        exit(1)

    windowName = "Webcam"
    cv2.namedWindow(windowName)

    key = -1
    while key == -1:        
        _, frame = camera.read()   
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = np.expand_dims(gray_frame, axis=-1)
        
        batch_input = np.expand_dims(frame, axis=0)
        batch_output = np.expand_dims(gray_frame, axis=0)
        
        batch_input = batch_input.astype("float32")/255.0
        batch_output = batch_output.astype("float32")/255.0
        
        losses = model.train_on_batch(batch_input, batch_output)
        print("LOSS:", losses)
        print("WEIGHTS:\n", filter_layer.weights[0].numpy())
        
        pred_image = model.predict(batch_input)
        pred_image = np.squeeze(pred_image, axis=0)
        pred_image *= 255.0
        pred_image = cv2.convertScaleAbs(pred_image)
        
        cv2.imshow("PREDICTION", pred_image)        
        
        cv2.imshow(windowName, frame)
        key = cv2.waitKey(30)

    camera.release()
    cv2.destroyAllWindows()
    
    print("IT'S OVER!!!!")

if __name__ == "__main__":
    main()
    

