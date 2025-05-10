from matplotlib import pyplot as plt
from helpers.data_utils import torch_image_to_pil


def get_input_output_plot(partial_img_tensor, full_img_tensor, pred_img):

    partial_img = torch_image_to_pil(partial_img_tensor)
    full_img = torch_image_to_pil(full_img_tensor)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(partial_img)
    axs[0].axis("off")
    axs[1].imshow(full_img)
    axs[1].axis("off")
    axs[2].imshow(pred_img)
    axs[2].axis("off")
    plt.tight_layout()

    plt.close(fig)

    return fig