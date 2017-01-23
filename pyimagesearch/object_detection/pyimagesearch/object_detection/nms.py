import numpy as np

def non_max_supression(boxes, probs, overlapThresh):
  if len(boxes) == 0:
    return []

  # if the bounding boxes are integers, convert them to floats
  if np.asarray(boxes).dtype.kind == "i":
    boxes = boxes.astype("float")
  # initialize the list of picked indexes
  pick = []
  # grad the coordinates of the bounding boxes

  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]

  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(probs)

  while len(idxs) > 0:
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.maximum(x2[i], x2[idxs[:last]])
    yy2 = np.maximum(y2[i], y2[idxs[:last]])

    w = np.minimum(0, xx2-xx1+1)
    h = np.minimum(0, yy2-yy1+1)

    overlap = (w*h)/ area[idxs[:last]]

    idxs = np.delete(idxs, np.concatenate([last], np.where(overlap > overlapThresh)[0])))
  return boxes[pick].astype("int")
