import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from torchvision.utils import save_image

from external.decalib.datasets import datasets
from external.decalib.deca import DECA
from external.decalib.models.FLAME import FLAME
from external.decalib.utils.config import cfg as deca_cfg
from external.decalib.utils.rotation_converter import batch_euler2axis, deg2rad
from util.landmark_image_generation import LandmarkImageGeneration
from util.util import save_coeffs, save_landmarks, save_params


def parse_args():
    """Configurations."""
    parser = argparse.ArgumentParser(description="test process of Face2FaceRHO")
    parser.add_argument("--device", default="cuda", type=str, help="set device, cpu for using cpu")
    parser.add_argument(
        "--src_img",
        type=str,
        default="val_case/000136_512.png",
        help="input source image (.jpg, .jpg, .jpeg, .png)",
    )
    # parser.add_argument(
    #     "--drv_img",
    #     type=str,
    #     default="val_case/driving/driving.jpg",
    #     help="input driving image (.jpg, .jpg, .jpeg, .png)",
    # )
    parser.add_argument(
        "--output_src_params",
        type=str,
        default=os.path.join("val_case", "source_136", "params.txt"),
        help="output head pose coefficients of source image (.txt)",
    )
    parser.add_argument(
        "--output_src_headpose",
        type=str,
        default=os.path.join("val_case", "source_136", "headpose.txt"),
        help="output head pose coefficients of source image (.txt)",
    )
    parser.add_argument(
        "--output_src_landmark",
        type=str,
        default=os.path.join("val_case", "source_136", "landmark.txt"),
        help="output facial landmarks of source image (.txt)",
    )
    parser.add_argument(
        "--output_src_landmark_imgs",
        type=str,
        default=os.path.join("val_case", "source_136", "landmark_imgs"),
        help="output facial landmarks of source image (.txt)",
    )
    parser.add_argument(
        "--output_front_params",
        type=str,
        default=os.path.join("val_case", "source_136_front", "params.txt"),
        help="output head pose coefficients of source image (.txt)",
    )
    parser.add_argument(
        "--output_front_headpose",
        type=str,
        default=os.path.join("val_case", "source_136_front", "headpose.txt"),
        help="output head pose coefficients of source image (.txt)",
    )
    parser.add_argument(
        "--output_front_landmark",
        type=str,
        default=os.path.join("val_case", "source_136_front", "landmark.txt"),
        help="output facial landmarks of source image (.txt)",
    )
    parser.add_argument(
        "--output_front_landmark_imgs",
        type=str,
        default=os.path.join("val_case", "source_136_front", "landmark_imgs"),
        help="output facial landmarks of source image (.txt)",
    )
    # parser.add_argument(
    #     "--output_drv_params",
    #     type=str,
    #     default=os.path.join("val_case", "driving_FLAME", "params.txt"),
    #     help="output head pose coefficients of source image (.txt)",
    # )
    # parser.add_argument(
    #     "--output_drv_headpose",
    #     type=str,
    #     default=os.path.join("val_case", "driving_FLAME", "headpose.txt"),
    #     help=" output head pose coefficients of driving image (.txt)",
    # )
    # parser.add_argument(
    #     "--output_drv_landmark",
    #     type=str,
    #     default=os.path.join("val_case", "driving_FLAME", "landmark.txt"),
    #     help="output driving facial landmarks (.txt, reconstructed by using shape coefficients "
    #     "of the source actor and expression and head pose coefficients of the driving actor)",
    # )
    # parser.add_argument(
    #     "--output_drv_landmark_imgs",
    #     type=str,
    #     default=os.path.join("val_case", "driving_FLAME", "landmark_imgs"),
    #     help="output facial landmarks of source image (.txt)",
    # )
    parser.add_argument(
        "--output_src_drv_headpose_params",
        type=str,
        default=os.path.join("val_case", "source_136_src_drv_headpose", "params.txt"),
        help="output head pose coefficients of source image (.txt)",
    )
    parser.add_argument(
        "--output_src_drv_headpose_headpose",
        type=str,
        default=os.path.join("val_case", "source_136_src_drv_headpose", "headpose.txt"),
        help="output head pose coefficients of source image (.txt)",
    )
    parser.add_argument(
        "--output_src_drv_headpose_landmark",
        type=str,
        default=os.path.join("val_case", "source_136_src_drv_headpose", "landmark.txt"),
        help="output facial landmarks of source image (.txt)",
    )
    parser.add_argument(
        "--output_src_drv_headpose_landmark_imgs",
        type=str,
        default=os.path.join("val_case", "source_136_src_drv_headpose", "landmark_imgs"),
        help="output facial landmarks of source image (.txt)",
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
        global_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].cuda()))
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
    args = parse_args()

    # 3DMM fitting by DECA: Detailed Expression Capture and Animation using FLAME model

    face_fitting = FLAMEFitting()
    torch.cuda.synchronize()
    start = time.perf_counter()
    src_params = face_fitting.fitting(args.src_img)
    # drv_params = face_fitting.fitting(args.drv_img)

    front_params = face_fitting.fitting_front(args.src_img)
    # src_drv_headpose_params = face_fitting.fitting_give_pose(args.src_img, drv_params["pose"])
    # front_params["ty"] = 0.0

    print("front_params pose:", front_params["pose"])

    pose_lml_extractor = PoseLandmarkExtractor()
    print("src_headpose")
    src_headpose = pose_lml_extractor.get_pose(
        src_params["shape"],
        src_params["exp"],
        src_params["pose"],
        src_params["scale"],
        src_params["tx"],
        src_params["ty"],
    )
    print("src_lmks")
    src_lmks = pose_lml_extractor.get_project_points(
        src_params["shape"],
        src_params["exp"],
        src_params["pose"],
        src_params["scale"],
        src_params["tx"],
        src_params["ty"],
    )
    print("front_headpose")
    front_headpose = pose_lml_extractor.get_pose(
        front_params["shape"],
        front_params["exp"],
        front_params["pose"],
        front_params["scale"],
        front_params["tx"],
        front_params["ty"],
    )
    print("front_lmks")
    front_lmks = pose_lml_extractor.get_project_points(
        front_params["shape"],
        front_params["exp"],
        front_params["pose"],
        front_params["scale"],
        front_params["tx"],
        front_params["ty"],
    )

    # src_drv_headpose_headpose = pose_lml_extractor.get_pose(
    #     src_drv_headpose_params["shape"],
    #     src_drv_headpose_params["exp"],
    #     src_drv_headpose_params["pose"],
    #     src_drv_headpose_params["scale"],
    #     src_drv_headpose_params["tx"],
    #     src_drv_headpose_params["ty"],
    # )

    # src_drv_headpose_lmks = pose_lml_extractor.get_project_points(
    #     src_drv_headpose_params["shape"],
    #     src_drv_headpose_params["exp"],
    #     src_drv_headpose_params["pose"],
    #     src_drv_headpose_params["scale"],
    #     src_drv_headpose_params["tx"],
    #     src_drv_headpose_params["ty"],
    # )

    # Note that the driving head pose and facial landmarks are calculated using the shape parameters of the source image
    # in order to eliminate the interference of the driving actor's identity.
    # drv_headpose = pose_lml_extractor.get_pose(
    #     src_params["shape"],
    #     drv_params["exp"],
    #     drv_params["pose"],
    #     drv_params["scale"],
    #     drv_params["tx"],
    #     drv_params["ty"],
    # )

    # drv_lmks = pose_lml_extractor.get_project_points(
    #     src_params["shape"],
    #     drv_params["exp"],
    #     drv_params["pose"],
    #     drv_params["scale"],
    #     drv_params["tx"],
    #     drv_params["ty"],
    # )
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"3DMM fitting time: {end - start}")

    # save
    os.makedirs(os.path.split(args.output_src_headpose)[0], exist_ok=True)
    save_coeffs(args.output_src_headpose, src_headpose)
    os.makedirs(os.path.split(args.output_src_landmark)[0], exist_ok=True)
    save_landmarks(args.output_src_landmark, src_lmks)
    os.makedirs(os.path.split(args.output_src_params)[0], exist_ok=True)
    save_params(args.output_src_params, src_params)

    # os.makedirs(os.path.split(args.output_drv_headpose)[0], exist_ok=True)
    # save_coeffs(args.output_drv_headpose, drv_headpose)
    # os.makedirs(os.path.split(args.output_drv_landmark)[0], exist_ok=True)
    # save_landmarks(args.output_drv_landmark, drv_lmks)
    # os.makedirs(os.path.split(args.output_drv_params)[0], exist_ok=True)
    # save_params(args.output_drv_params, drv_params)

    os.makedirs(os.path.split(args.output_front_headpose)[0], exist_ok=True)
    save_coeffs(args.output_front_headpose, front_headpose)
    os.makedirs(os.path.split(args.output_front_landmark)[0], exist_ok=True)
    save_landmarks(args.output_front_landmark, front_lmks)
    os.makedirs(os.path.split(args.output_front_params)[0], exist_ok=True)
    save_params(args.output_front_params, front_params)

    # os.makedirs(os.path.split(args.output_src_drv_headpose_headpose)[0], exist_ok=True)
    # save_coeffs(args.output_src_drv_headpose_headpose, src_drv_headpose_headpose)
    # os.makedirs(os.path.split(args.output_src_drv_headpose_landmark)[0], exist_ok=True)
    # save_landmarks(args.output_src_drv_headpose_landmark, src_drv_headpose_lmks)
    # os.makedirs(os.path.split(args.output_src_drv_headpose_params)[0], exist_ok=True)
    # save_params(args.output_src_drv_headpose_params, src_drv_headpose_params)

    src_lmks = torch.from_numpy(np.array(src_lmks)).float()
    # drv_lmks = torch.from_numpy(np.array(drv_lmks)).float()
    front_lmks = torch.from_numpy(np.array(front_lmks)).float()
    # src_drv_headpose_lmks = torch.from_numpy(np.array(src_drv_headpose_lmks)).float()

    landmark_img_generator = LandmarkImageGeneration(512)
    src_landmark_imgs = landmark_img_generator.generate_landmark_img(src_lmks)
    # drv_landmark_imgs = landmark_img_generator.generate_landmark_img(drv_lmks)
    front_landmark_imgs = landmark_img_generator.generate_landmark_img(front_lmks)
    # src_drv_headpose_landmark_imgs = landmark_img_generator.generate_landmark_img(src_drv_headpose_lmks)

    os.makedirs(args.output_src_landmark_imgs, exist_ok=True)
    for i, src_lmk_img in enumerate(src_landmark_imgs):
        save_image(src_lmk_img, args.output_src_landmark_imgs + f"/{i}.png")

    # os.makedirs(args.output_drv_landmark_imgs, exist_ok=True)
    # for i, drv_lmk_img in enumerate(drv_landmark_imgs):
    #     save_image(drv_lmk_img, args.output_drv_landmark_imgs + f"/{i}.png")

    os.makedirs(args.output_front_landmark_imgs, exist_ok=True)
    for i, front_lmk_img in enumerate(front_landmark_imgs):
        save_image(front_lmk_img, args.output_front_landmark_imgs + f"/{i}.png")


    # os.makedirs(args.output_src_drv_headpose_landmark_imgs, exist_ok=True)
    # for i, src_drv_headpose_lmk_img in enumerate(src_drv_headpose_landmark_imgs):
    #     save_image(src_drv_headpose_lmk_img, args.output_src_drv_headpose_landmark_imgs + f"/{i}.png")
