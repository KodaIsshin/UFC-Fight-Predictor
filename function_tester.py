import torch

prediction = .4
predict= torch.tensor(prediction)
predict =  (predict > 0.5).float()
print(predict)