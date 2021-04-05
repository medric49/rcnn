import config
import yorutorch
import nets
import cv2
import torch
if __name__ == '__main__':
    image = cv2.imread('daenerys.jpg')
    image = cv2.resize(image, (config.im_width, config.im_height))

    image = [image]

    image = torch.tensor(image, dtype=torch.float).cuda()
    image = image.view((1, 3, config.im_height, config.im_width))

    net = nets.DCNNet()
    model = yorutorch.models.Model(net).to(yorutorch.devices.cuda_otherwise_cpu)
    print(net)
    output = model(image)
    print(output.shape)
