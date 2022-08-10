import argparse
import copy
import json
import os

import cv2
import numpy as np
import torch

from torchvision.utils import save_image
from tqdm import tqdm

from external.decalib.datasets import datasets
from external.decalib.deca import DECA
from external.decalib.models.FLAME import FLAME
from external.decalib.utils.config import cfg as deca_cfg
from external.decalib.utils.rotation_converter import batch_euler2axis, deg2rad
from models import create_model
from options.parse_config import Face2FaceRHOConfigParse
from util.landmark_image_generation import LandmarkImageGeneration
from util.util import read_target, tensor2im


def parse_args():
    """Configurations."""
    parser = argparse.ArgumentParser(description="test process of Face2FaceRHO")
    parser.add_argument("--device", default="cuda", type=str, help="set device, cpu for using cpu")
    parser.add_argument("--config", type=str, default="src/config/test_face2facerho.ini", help=".ini config file name")
    parser.add_argument(
        "--src_video_dir",
        type=str,
        default="val_case_video_512",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="val_case_video_512_results_recons",
    )

    return _check_args(parser.parse_args())


def _check_args(args):
    if args is None:
        raise RuntimeError("Invalid arguments!")
    return args


class FLAMEFitting:
    def __init__(self):
        self.deca = DECA(config=deca_cfg, device=args.device)

    def fitting(self, img_name):
        testdata = datasets.TestData(img_name, iscrop=False, face_detector="fan", sample_step=10)
        input_data = testdata[0]
        images = input_data["image"].to(args.device)[None, ...]
        with torch.no_grad():
            codedict = self.deca.encode(images)
            codedict["tform"] = input_data["tform"][None, ...]
            original_image = input_data["original_image"][None, ...]
            _, _, h, w = original_image.shape
            params = self.deca.ensemble_3DMM_params(
                codedict, image_size=deca_cfg.dataset.image_size, original_image_size=h
            )

        return params

    def fitting_front(self, img_name):
        testdata = datasets.TestData(img_name, iscrop=False, face_detector="fan", sample_step=10)
        input_data = testdata[0]
        images = input_data["image"].to(args.device)[None, ...]
        euler_pose = torch.zeros((1, 3))
        global_pose = batch_euler2axis(deg2rad(euler_pose[:, :3].cuda()))
        with torch.no_grad():
            codedict = self.deca.encode(images)
            codedict["pose"][:, 1] = global_pose[:, 1]
            codedict["pose"][:, 2] = global_pose[:, 2]
            # codedict["cam"][:, :] = 0.0
            # codedict["cam"][:, 0] = 8
            codedict["tform"] = input_data["tform"][None, ...]
            original_image = input_data["original_image"][None, ...]
            _, _, h, w = original_image.shape
            params = self.deca.ensemble_3DMM_params(
                codedict, image_size=deca_cfg.dataset.image_size, original_image_size=h
            )

        return params


class PoseLandmarkExtractor:
    def __init__(self):
        self.flame = FLAME(deca_cfg.model)

        with open(os.path.join(deca_cfg.deca_dir, "data", "pose_transform_config.json"), "r") as f:
            pose_transform = json.load(f)

        self.scale_transform = pose_transform["scale_transform"]
        self.tx_transform = pose_transform["tx_transform"]
        self.ty_transform = pose_transform["ty_transform"]
        self.tx_scale = 0.256  # 512 / 2000
        self.ty_scale = -self.tx_scale

    @staticmethod
    def transform_points(points, scale, tx, ty):
        trans_matrix = torch.zeros((1, 4, 4), dtype=torch.float32)
        trans_matrix[:, 0, 0] = scale
        trans_matrix[:, 1, 1] = -scale
        trans_matrix[:, 2, 2] = 1
        trans_matrix[:, 0, 3] = tx
        trans_matrix[:, 1, 3] = ty
        trans_matrix[:, 3, 3] = 1

        batch_size, n_points, _ = points.shape
        points_homo = torch.cat([points, torch.ones([batch_size, n_points, 1], dtype=points.dtype)], dim=2)
        points_homo = points_homo.transpose(1, 2)
        trans_points = torch.bmm(trans_matrix, points_homo).transpose(1, 2)
        trans_points = trans_points[:, :, 0:3]
        return trans_points

    def get_project_points(self, shape_params, expression_params, pose, scale, tx, ty):
        shape_params = torch.tensor(shape_params).unsqueeze(0)
        expression_params = torch.tensor(expression_params).unsqueeze(0)
        pose = torch.tensor(pose).unsqueeze(0)
        verts, landmarks3d = self.flame(
            shape_params=shape_params, expression_params=expression_params, pose_params=pose
        )
        trans_landmarks3d = self.transform_points(landmarks3d, scale, tx, ty)
        trans_landmarks3d = trans_landmarks3d.squeeze(0).cpu().numpy()
        return trans_landmarks3d[:, 0:2].tolist()

    def calculate_nose_tip_tx_ty(self, shape_params, expression_params, pose, scale, tx, ty):
        front_pose = copy.deepcopy(pose)
        front_pose[0] = front_pose[1] = front_pose[2] = 0
        front_landmarks3d = self.get_project_points(shape_params, expression_params, front_pose, scale, tx, ty)
        original_landmark3d = self.get_project_points(shape_params, expression_params, pose, scale, tx, ty)
        nose_tx = original_landmark3d[30][0] - front_landmarks3d[30][0]
        nose_ty = original_landmark3d[30][1] - front_landmarks3d[30][1]
        return nose_tx, nose_ty

    def get_pose(self, shape_params, expression_params, pose, scale, tx, ty):
        nose_tx, nose_ty = self.calculate_nose_tip_tx_ty(shape_params, expression_params, pose, scale, tx, ty)
        transformed_axis_angle = [float(pose[0]), float(pose[1]), float(pose[2])]
        transformed_tx = self.tx_transform + self.tx_scale * (tx + nose_tx)
        transformed_ty = self.ty_transform + self.ty_scale * (ty + nose_ty)
        transformed_scale = scale / self.scale_transform
        return transformed_axis_angle + [transformed_tx, transformed_ty, transformed_scale]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    args = parse_args()

    # 3DMM fitting by DECA: Detailed Expression Capture and Animation using FLAME model

    face_fitting = FLAMEFitting()
    pose_lml_extractor = PoseLandmarkExtractor()
    config_parse = Face2FaceRHOConfigParse()
    landmark_img_generator = LandmarkImageGeneration(512)
    opt = config_parse.get_opt_from_ini(args.config)
    config_parse.setup_environment()

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    img_files = sorted(os.listdir(args.src_video_dir))
    first_img_name = img_files[0]
    first_img_name = os.path.join(args.src_video_dir, first_img_name)

    src_params = face_fitting.fitting(first_img_name)
    src_headpose = pose_lml_extractor.get_pose(
        src_params["shape"],
        src_params["exp"],
        src_params["pose"],
        src_params["scale"],
        src_params["tx"],
        src_params["ty"],
    )
    src_lmks = pose_lml_extractor.get_project_points(
        src_params["shape"],
        src_params["exp"],
        src_params["pose"],
        src_params["scale"],
        src_params["tx"],
        src_params["ty"],
    )

    for img_file in tqdm(img_files[1:3]):
        img_path = os.path.join(args.src_video_dir, img_file)
        output_file_name = os.path.join(args.output_dir, img_file)
        # src_params = face_fitting.fitting(img_path)
        drv_params = face_fitting.fitting(img_path)

        # src_headpose = pose_lml_extractor.get_pose(
        #     src_params["shape"],
        #     src_params["exp"],
        #     src_params["pose"],
        #     src_params["scale"],
        #     src_params["tx"],
        #     src_params["ty"],
        # )
        # src_lmks = pose_lml_extractor.get_project_points(
        #     src_params["shape"],
        #     src_params["exp"],
        #     src_params["pose"],
        #     src_params["scale"],
        #     src_params["tx"],
        #     src_params["ty"],
        # )
        drv_headpose = pose_lml_extractor.get_pose(
            src_params["shape"],
            drv_params["exp"],
            drv_params["pose"],
            drv_params["scale"],
            drv_params["tx"],
            drv_params["ty"],
        )
        drv_lmks = pose_lml_extractor.get_project_points(
            src_params["shape"],
            drv_params["exp"],
            drv_params["pose"],
            drv_params["scale"],
            drv_params["tx"],
            drv_params["ty"],
        )

        src_lmks = torch.from_numpy(np.array(src_lmks)).float()
        src_headpose = torch.from_numpy(np.array(src_headpose)).float()
        drv_lmks = torch.from_numpy(np.array(drv_lmks)).float()
        drv_headpose = torch.from_numpy(np.array(drv_headpose)).float()

        src_landmark_imgs = landmark_img_generator.generate_landmark_img(src_lmks)
        drv_landmark_imgs = landmark_img_generator.generate_landmark_img(drv_lmks)

        src_face = dict()
        src_face["img"] = read_target(img_path, 512)
        src_face["landmarks"] = src_lmks
        src_face["headpose"] = src_headpose
        src_face["landmark_img"] = [v.unsqueeze(0) for v in src_landmark_imgs]

        drv_face = dict()
        drv_face["landmarks"] = drv_lmks
        drv_face["headpose"] = drv_headpose
        drv_face["landmark_img"] = [v.unsqueeze(0) for v in drv_landmark_imgs]

        model.set_source_face(src_face["img"].unsqueeze(0), src_face["headpose"].unsqueeze(0))

        model.reenactment(src_face["landmark_img"], drv_face["headpose"].unsqueeze(0), drv_face["landmark_img"])

        visual_results = model.get_current_visuals()

        im = tensor2im(visual_results["fake"])
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_file_name, im)

    # def visualize_link(img, annotation, output_path, line_type="-*"):
    #     """
    #     visualize the linked facial landmarks according to their physical locations
    #     """
    #     # plt.figure()
    #     plt.imshow(img)  # show face image
    #     x = np.array(annotation[:, 0])
    #     y = np.array(annotation[:, 1])
    #     star = line_type  # plot style, such as '-*'
    #     # plt.plot(x[0:9], y[0:9], star)  # face contour
    #     # plt.plot(x[9:17], y[9:17], star)  # face contour
    #     # plt.plot(x[17:22], y[17:22], star)
    #     # plt.plot(x[22:27], y[22:27], star)
    #     # plt.plot(x[27:31], y[27:31], star)
    #     # plt.plot(x[31:36], y[31:36], star)
    #     # plt.plot(np.hstack([x[36:44], x[36]]), np.hstack([y[36:44], y[36]]), star)
    #     # plt.plot(np.hstack([x[44:52], x[44]]), np.hstack([y[44:52], y[44]]), star)
    #     # plt.plot(np.hstack([x[52:64], x[52]]), np.hstack([y[52:64], y[52]]), star)
    #     # plt.plot(np.hstack([x[64:72], x[64]]), np.hstack([y[64:72], y[64]]), star)
    #     plt.axis("off")
    #     plt.savefig(output_path)

    # def crop_square(img, size=256, interpolation=cv2.INTER_AREA):
    #     h, w = img.shape[:2]
    #     min_size = np.amin([h, w])

    #     # Centralize and crop
    #     crop_img = img[
    #         int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
    #         int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    #     ]
    #     resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    #     return resized

    # image_1 = cv2.imread(first_img_name)
    # image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    # image_1 = crop_square(image_1, 512)

    # print(src_lmks)
    # src_lmks_tensor = torch.from_numpy(np.array(src_lmks)).float()
    # src_lmk_imgs = landmark_img_generator.generate_landmark_img(src_lmks_tensor)

    # for i, src_lmk_img in enumerate(src_lmk_imgs):
    #     save_image(src_lmk_img, f"{i}.png")

    # annotation = np.array(src_lmks)
    # annotation = (annotation + 1) / 2 * 512
    # print("annotation shape", annotation.shape)
    # visualize_link(image_1, annotation, "vis_link.png")
    # print(annotation)
