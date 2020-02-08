import cv2
import numpy as np
import matplotlib.pyplot as plt

# If you are using your own custom image, follow the format below for your ground truth file:
# [int - represents the number of faces in the image]
# [x1, y1, w, h] <---- face 1
# [x1, y1, w, h] <---- face 2
# ...
# [x1, y1, w, h] <---- face 3

def use_custom(custom_gt: str):
  print("Reading ground truth...")
  results = []
  out = []
  with open(custom_gt, "r") as file:
    results = file.readlines()

  for result in results:
    cvt_result = result.split(" ")
    bounding_box = list(map(int, cvt_result))
    out.append(bounding_box)

  return out

def generate_gt(results, name = 'gt.txt'):
  # Write out the coordinates to file:
  print('Writing out ground truth with name: {0}'.format(name))
  with open(name, "w+") as file:
    for result in results:
      bbox = result['box']
      file.write("%i %i %i %i\n"%(bbox[0], bbox[1], bbox[2], bbox[3]))


def run_diagnostic(image_path, detector):
    #test image:
    #image_path = "./WIDER_val/images/0--Parade/0_Parade_marchingband_1_234.jpg"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Input image: {0}".format(image_path))

    plt.figure(figsize=(10, 15))
    plt.imshow(img)
    plt.grid(None)
    plt.axis('off')

    print("Detecting...")
    results = detector.detect_faces(img)

    rectColor = (255,0,0)
    for result in results:
      bounding_box = result['box']
      cv2.rectangle(img,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  rectColor,
                  2)

    plt.figure(figsize=(10, 15))
    plt.imshow(img)
    plt.grid(None)
    plt.axis('off')

    print("Detected: {0}".format(len(results)))

def run_noise_function_example(IMG_VAR, uniform_noise, detector):
    org_img = cv2.imread(IMG_VAR)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    org_img = uniform_noise(org_img)
    results = detector.detect_faces(org_img)

    rectColor = (255,0,0)

    for result in results:
      bounding_box = result['box']
      cv2.rectangle(org_img,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  rectColor,
                  2)

    plt.figure(figsize=(10, 15))
    plt.imshow(org_img)
    plt.grid(None)
    plt.axis('off')
    print("Detected {0} faces".format(len(results)))

def show_gt(org_img, USE_WIDER, ):
    # Display ground truths:
    gr_img = org_img

    rectColor = (0,255,0)
    for truth in gt:
      if(USE_WIDER):
          bounding_box = truth.astype(int)
      else:
          bounding_box = truth
      cv2.rectangle(gr_img,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  rectColor,
                  3)
    
    print("Ground Truths visualized")
    print("Total face count in grouth truth: ", len(gt))
    plt.figure(figsize=(10, 15))
    plt.imshow(gr_img)
    plt.grid(None)
    plt.axis('off')

def generateInteractiveGraphInst(AP_SCORES, imgList, result_pool, i: int, savingMode=False):
    currImg = imgList[i]
    rectColor = (0,255,0)
    for result in result_pool[i]:
        bounding_box = result['box']
        cv2.rectangle(currImg,
                    (bounding_box[0], bounding_box[1]),
                    (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                    rectColor,
                    3)

    plt.figure(figsize=(9,9))
    plt.subplot(2, 2, 1)
    plt.title('Img Noise Intensity: %s' % i)
    plt.imshow(currImg)
    plt.subplot(2, 2, 2)
    plt.title('AP Graph Performance: {}%'.format(AP_SCORES[i]))
    plt.xlim(i-35, i+35)
    plt.plot(AP_SCORES, '-o', markevery=[i])
    plt.tight_layout()

    if(savingMode == False):
        plt.show()

    if(savingMode):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.ioff()
        return buf
    else:
        return None
