## RUN FILE normally -> python demo.py 

from click import argument
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import gradio as gr
import json

def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		for x in range(0, numCols):
			if scoresData[x] < args["min_confidence"]:
				continue
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	return (rects, confidences)

def predict(image, arguments):
    # image - is an array
    # arguments - will be a string containing dict, e.g -> "{'a':1, 'b': 2}"
    args = json.loads(arguments)  # this will convert the string into dict to use it afterwards
    orig = image.copy()
    (origH, origW) = image.shape[:2]
    (newW, newH) = (args["width"], args["height"])
    rW = origW / float(newW)
    rH = origH / float(newH)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
#    Layer Names
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    # print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    results = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        dX = int((endX - startX) * args["padding"])
        dY = int((endY - startY) * args["padding"])
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        roi = orig[startY:endY, startX:endX]
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)
        results.append(((startX, startY, endX, endY), text))
    results = sorted(results, key=lambda r:r[0][1])
    for ((startX, startY, endX, endY), text) in results:
        # print("OCR TEXT")
        # print("========")
        # print("{}\n".format(text))
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return output

with gr.Blocks() as demo:
    args = {}
    gr.Markdown("## OCR")
    with gr.Row():
        args['east'] = gr.Textbox(placeholder="path to input EAST text detector", label = "EAST")
        args['min_confidence'] = gr.Number(label = "min-confidence") 
    with gr.Row():
        args['width'] = gr.Number(label = "width", precision=0) 
        args['height'] = gr.Number(label = "height", precision=0) 
        args['padding'] = gr.Number(label = "padding") 

    with gr.Row():
        im_1 = gr.Image()
        im_2 = gr.Image(label = "Output")
    
    btn = gr.Button(value="Submit")

    # here we can pass the args as input to the below button as string -> str(args)
    btn.click(predict, inputs=[im_1]#, str(args)]
    , outputs=[im_2])

if __name__ == "__main__":
    demo.launch()

#image = cv2.imread(args["image"])
#Image Input from which text to read
