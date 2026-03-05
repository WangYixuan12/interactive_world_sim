import os

import cv2
import numpy as np


class DataSampler:
    """Data sampler"""

    def __init__(self, img_h: int, img_w: int, T_length: int, task: str):
        self.img_h = img_h
        self.img_w = img_w
        self.T_length = T_length
        self.T_width = T_length / 4.0
        self.size = img_h / 6.0
        self.eef_rand_left = img_w / 4.0
        self.task = task

    def sample_T_image(self, text: str) -> np.ndarray:
        """Sample image"""
        T_x = np.random.randint(
            self.img_h / 2.0 - self.size, self.img_h / 2.0 + self.size
        )
        T_y = np.random.randint(
            self.img_w / 2.0 - self.size, self.img_w / 2.0 + self.size
        )
        rot_angle = np.random.uniform(0.0, np.pi * 2)
        img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        half_width = self.T_width / 2.0

        horizontal = np.array(
            [
                [-self.T_length / 2.0, -half_width],
                [self.T_length / 2.0, -half_width],
                [self.T_length / 2.0, half_width],
                [-self.T_length / 2.0, half_width],
            ],
            dtype=np.float32,
        )
        vertical = np.array(
            [
                [-half_width, half_width],
                [half_width, half_width],
                [half_width, self.T_length * 3.0 / 4.0 + half_width],
                [-half_width, self.T_length * 3.0 / 4.0 + half_width],
            ],
            dtype=np.float32,
        )

        def to_image_coords(points: np.ndarray) -> np.ndarray:
            transformed = points @ rotation.T
            transformed[:, 0] += T_y
            transformed[:, 1] += T_x
            return np.round(transformed).astype(np.int32)

        cv2.fillConvexPoly(img, to_image_coords(horizontal), 255)
        cv2.fillConvexPoly(img, to_image_coords(vertical), 255)

        left_eef_x = np.random.randint(0.0, self.eef_rand_left)
        left_eef_y = np.random.randint(self.img_h / 4.0, self.img_h * 3.0 / 4.0)
        right_eef_x = np.random.randint(self.img_w - self.eef_rand_left, self.img_w)
        right_eef_y = np.random.randint(self.img_h / 4.0, self.img_h * 3.0 / 4.0)
        cv2.circle(img, (left_eef_x, left_eef_y), self.img_h // 20, 255, -1)
        cv2.circle(img, (right_eef_x, right_eef_y), self.img_h // 20, 255, -1)

        cv2.putText(
            img,
            text,
            (10, self.img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,),
            thickness=3,
        )
        img = np.tile(img[:, :, None], (1, 1, 3))
        img = img.astype(np.uint8)

        return img

    def sample_rgb_block(self, idx: int) -> np.ndarray:
        """Sample texts R, G, B positions to indicate block locations."""
        R_pos = (
            np.random.randint(self.img_w * 1.0 / 3.0, self.img_w * 2.0 / 3.0),
            np.random.randint(self.img_h * 1.0 / 3.0, self.img_h * 2.0 / 3.0),
        )
        G_pos = (
            np.random.randint(self.img_w * 1.0 / 3.0, self.img_w * 2.0 / 3.0),
            np.random.randint(self.img_h * 1.0 / 3.0, self.img_h * 2.0 / 3.0),
        )
        B_pos = (
            np.random.randint(self.img_w * 1.0 / 3.0, self.img_w * 2.0 / 3.0),
            np.random.randint(self.img_h * 1.0 / 3.0, self.img_h * 2.0 / 3.0),
        )
        img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        cv2.putText(
            img,
            "R",
            R_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            thickness=5,
        )
        cv2.putText(
            img,
            "G",
            G_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            thickness=5,
        )
        cv2.putText(
            img,
            "B",
            B_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            thickness=5,
        )
        # case 0: sweep all in
        # case 1: sweep R out
        # case 2: sweep G out
        # case 3: sweep B out
        # case 4: sweep R, G out
        # case 5: sweep R, B out
        # case 6: sweep G, B out
        # case 7: sweep all out
        case = idx % 8
        if case == 0:
            text = "sweep all in"
        elif case == 1:
            text = "sweep R out"
        elif case == 2:
            text = "sweep G out"
        elif case == 3:
            text = "sweep B out"
        elif case == 4:
            text = "sweep R, G out"
        elif case == 5:
            text = "sweep R, B out"
        elif case == 6:
            text = "sweep G, B out"
        elif case == 7:
            text = "sweep all out"
        cv2.putText(
            img,
            text,
            (10, self.img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            thickness=3,
        )
        return img

    def sample_T(self, idx: int) -> np.ndarray:
        """Sample an image based on the given index."""
        if idx % 10 == 0:
            text = "random no contact"
        elif idx % 10 == 1:
            text = "random contact"
        elif idx % 10 == 2:
            text = "push right"
        elif idx % 10 == 3:
            text = "push left"
        elif idx % 10 == 4:
            text = "push up"
        elif idx % 10 == 5:
            text = "push down"
        elif idx % 10 >= 6 and idx % 10 <= 7:
            text = "rotate clockwise"
        elif idx % 10 >= 8 and idx % 10 <= 9:
            text = "rotate counterclockwise"
        return self.sample_T_image(text)

    def sample_rope(self, idx: int) -> np.ndarray:
        """Sample an image based on the given index."""
        if idx % 4 == 0:
            text = "in 1, in 2"
        elif idx % 4 == 1:
            text = "in 1, out 2"
        elif idx % 4 == 2:
            text = "out 1, in 2"
        elif idx % 4 == 3:
            text = "out 1, out 2"
        init_config = np.random.randint(0, 3)
        if init_config == 0:
            text += ", left loop"
        elif init_config == 1:
            text += ", middle loop"
        elif init_config == 2:
            text += ", bottom loop"
        mask = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        cv2.putText(
            mask,
            text,
            (10, self.img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            thickness=3,
        )
        return mask

    def sample_bimanual_rope(self, idx: int) -> np.ndarray:
        """Sample an image based on the given index."""
        if idx % 4 == 0:
            text = "in 1, in 2"
        elif idx % 4 == 1:
            text = "in 1, out 2"
        elif idx % 4 == 2:
            text = "out 1, in 2"
        elif idx % 4 == 3:
            text = "out 1, out 2"
        init_config = np.random.randint(0, 3)
        if init_config == 0:
            text += ", top loop"
        elif init_config == 1:
            text += ", middle loop"
        elif init_config == 2:
            text += ", bottom loop"
        mask = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        cv2.putText(
            mask,
            text,
            (10, self.img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            thickness=3,
        )
        return mask

    def sample_single_grasp(self, idx: int) -> np.ndarray:
        """Sample an image based on the given index."""
        # sample the position: plate on left and the cup on right
        # sample placing loc
        # sample cup bar (-45~45)
        # play data mode
        # plate_x = np.random.randint(self.img_w // 4, self.img_w // 2)
        # plate_y = np.random.randint(self.img_h // 4, self.img_h * 3 // 4)
        # cup_x = np.random.randint(self.img_w // 2, self.img_w * 3 // 4)
        # cup_y = np.random.randint(self.img_h // 4, self.img_h * 3 // 4)
        # place_x = plate_x + np.random.randint(-self.img_w // 8, self.img_w // 8)
        # place_y = plate_y + np.random.randint(-self.img_h // 8, self.img_h // 8)
        # bar_angle = np.random.randint(-45, 45)

        # # demo data mode
        # plate_x = np.random.randint(
        #     self.img_w // 2 - self.img_w // 8, self.img_w // 2
        # )
        # plate_y = np.random.randint(
        #     self.img_h // 2 - self.img_h // 8, self.img_h // 2 + self.img_h // 8
        # )
        # cup_x = np.random.randint(self.img_w // 2, self.img_w // 2 + self.img_w // 8)
        # cup_y = np.random.randint(
        #     self.img_h // 2 - self.img_h // 8, self.img_h // 2 + self.img_h // 8
        # )
        # place_x = plate_x
        # place_y = plate_y
        # bar_angle = np.random.randint(-10, 10)

        # demo data mode v2
        plate_x = np.random.randint(self.img_w // 2 - self.img_w // 12, self.img_w // 2)
        plate_y = np.random.randint(
            self.img_h // 2 - self.img_h // 8, self.img_h // 2 + self.img_h // 8
        )
        cup_x = np.random.randint(
            self.img_w // 2 + self.img_w // 12, self.img_w // 2 + self.img_w // 8
        )
        cup_y = np.random.randint(
            self.img_h // 2 - self.img_h // 8, self.img_h // 2 + self.img_h // 8
        )
        place_x = plate_x
        place_y = plate_y
        bar_angle = np.random.randint(-10, 10)

        img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        cv2.circle(img, (plate_x, plate_y), self.img_h // 10, (0, 255, 0), -1)
        cv2.circle(img, (cup_x, cup_y), self.img_h // 20, (255, 0, 0), -1)
        cv2.circle(img, (place_x, place_y), self.img_h // 20, (0, 0, 255), 2)
        cv2.line(
            img,
            (cup_x, cup_y),
            (
                int(cup_x + 50 * np.cos(np.deg2rad(bar_angle))),
                int(cup_y - 50 * np.sin(np.deg2rad(bar_angle))),
            ),
            (255, 0, 0),
            3,
        )

        # sample the picking action pattern
        # case 0: rotate x (0~90) degree up and pick
        # case 1: rotate x (0~90) degree down and pick
        # case 2: directly pick
        # case 3: failed gripper closure
        # case 4: perturb after placing
        if idx % 5 == 0:
            text = f"rotate up {np.random.randint(0, 91)} and pick"
        elif idx % 5 == 1:
            text = f"rotate down {np.random.randint(0, 91)} and pick"
        elif idx % 5 == 2:
            text = "directly pick"
        elif idx % 5 == 3:
            text = "failed gripper closure"
        elif idx % 5 == 4:
            text = "perturb after placing"
        cv2.putText(
            img,
            text,
            (10, self.img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            thickness=3,
        )
        return img

    def sample_box(self, idx: int) -> np.ndarray:
        """Sample an image based on the given index."""
        box_x = np.random.randint(self.img_w * 3.0 / 8.0, self.img_w * 5.0 / 8.0)
        box_y = np.random.randint(self.img_h / 3.0, self.img_h / 2.0)
        rot_angle = np.random.uniform(np.pi * 3.0 / 8.0, np.pi * 5.0 / 8.0)
        img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255

        def draw_oriented_rectangle(
            image: np.ndarray,
            center: tuple,
            size: tuple,
            angle: float,
            color: tuple,
            thickness: int,
        ) -> np.ndarray:
            """Draws an oriented (rotated) rectangle on an image.

            :param image: The input image (NumPy array).
            :param center: A tuple (x, y) for the center of the rectangle.
            :param size: A tuple (width, height) for the size of the rectangle.
            :param angle: The rotation angle in degrees (clockwise).
            :param color: The color in BGR format (e.g., (0, 255, 0) for green).
            :param thickness: The line thickness (e.g., 2 for outline, -1 for filled).
            """
            # Create the RotatedRect structure: (center, size, angle)
            rect = (center, size, angle)

            # Get the four corner vertices of the rotated rectangle
            box = cv2.boxPoints(rect)

            # Convert the floating-point vertices to integers
            box = np.int0(box)

            # Draw the contours (the polygon defined by the vertices)
            cv2.drawContours(image, [box], 0, color, thickness)

            # Draw the line representing the opening edge
            cv2.line(image, tuple(box[0]), tuple(box[1]), (0, 0, 255), thickness)

            return image

        img = draw_oriented_rectangle(
            img,
            (box_x, box_y),
            (100, 100),
            np.rad2deg(rot_angle),
            (0, 255, 0),
            5,
        )

        # sample the placing action pattern
        # case 0: place inside box
        # case 1: place outside box
        if idx % 4 == 0:
            text = "place inside box"
        elif idx % 4 == 1:
            text = "open box"
        elif idx % 4 == 2:
            text = "place outside box"
        elif idx % 4 == 3:
            text = "open box"
        cv2.putText(
            img,
            text,
            (10, self.img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            thickness=3,
        )
        return img

    def sample_chain_in_box(self, idx: int) -> np.ndarray:
        """Sample an image based on the given index."""
        # # play mode
        # box_x = np.random.randint(self.img_w * 3.0 / 8.0, self.img_w * 4.0 / 8.0)
        # box_y = np.random.randint(self.img_h / 3.0, self.img_h * 2.0 / 3.0)
        # rot_angle = np.random.uniform(np.pi * 3.0 / 8.0, np.pi * 5.0 / 8.0)

        # demo mode
        box_x = np.random.randint(
            self.img_w / 2.0 - self.img_w / 12.0, self.img_w / 2.0
        )
        box_y = np.random.randint(
            self.img_h / 2.0 - self.img_h / 12.0, self.img_h / 2.0 + self.img_h / 12.0
        )
        rot_angle = np.random.uniform(
            np.pi / 2.0 - np.pi / 12.0, np.pi / 2.0 + np.pi / 12.0
        )

        img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255

        def draw_oriented_rectangle(
            image: np.ndarray,
            center: tuple,
            size: tuple,
            angle: float,
            color: tuple,
            thickness: int,
        ) -> np.ndarray:
            """Draws an oriented (rotated) rectangle on an image.

            :param image: The input image (NumPy array).
            :param center: A tuple (x, y) for the center of the rectangle.
            :param size: A tuple (width, height) for the size of the rectangle.
            :param angle: The rotation angle in degrees (clockwise).
            :param color: The color in BGR format (e.g., (0, 255, 0) for green).
            :param thickness: The line thickness (e.g., 2 for outline, -1 for filled).
            """
            # Create the RotatedRect structure: (center, size, angle)
            rect = (center, size, angle)

            # Get the four corner vertices of the rotated rectangle
            box = cv2.boxPoints(rect)

            # Convert the floating-point vertices to integers
            box = np.int0(box)

            # Draw the contours (the polygon defined by the vertices)
            cv2.drawContours(image, [box], 0, color, thickness)

            # Draw the line representing the opening edge
            cv2.line(image, tuple(box[0]), tuple(box[1]), (0, 0, 255), thickness)

            return image

        img = draw_oriented_rectangle(
            img,
            (box_x, box_y),
            (100, 100),
            np.rad2deg(rot_angle),
            (0, 255, 0),
            5,
        )

        # sample the placing action pattern
        # case 0: place inside box
        # case 1: place outside box
        if idx % 3 == 0:
            text = "place inside box"
        elif idx % 3 == 1:
            text = "place outside box"
        elif idx % 3 == 2:
            text = "failed grasping"
        cv2.putText(
            img,
            text,
            (10, self.img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            thickness=3,
        )
        return img

    def sample(self, idx: int) -> np.ndarray:
        """Sample an image based on the given index."""
        if self.task == "bimanual_push":
            return self.sample_T(idx)
        elif self.task == "single_rope":
            return self.sample_rope(idx)
        elif self.task == "bimanual_rope":
            return self.sample_bimanual_rope(idx)
        elif self.task == "bimanual_sweep":
            return self.sample_rgb_block(idx)
        elif self.task == "bimanual_sweep_v2":
            return self.sample_rgb_block(idx)
        elif self.task == "single_grasp":
            return self.sample_single_grasp(idx)
        elif self.task == "bimanual_box":
            return self.sample_box(idx)
        elif self.task == "single_chain_in_box":
            return self.sample_chain_in_box(idx)
        return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)


def generate_data_samples(task: str, num_samples: int, output_dir: str) -> None:
    data_sampler = DataSampler(img_h=480, img_w=640, T_length=200, task=task)
    os.system(f"mkdir -p {output_dir}")
    i = 0
    while True:
        img = data_sampler.sample(idx=i)
        cv2.imshow("sampled_image", img)
        cv2.waitKey(30)
        good = input(f"Sample {i} generated. Accept? (y/n): ")
        if good.lower() == "y":
            i += 1
            cv2.imwrite(f"{output_dir}/sample_{i}.png", img)
        else:
            print("Resampling...")
        if i >= num_samples:
            break


if __name__ == "__main__":
    generate_data_samples(
        task="single_grasp", num_samples=10, output_dir="data/eval_init/single_grasp"
    )
