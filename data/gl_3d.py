from manimlib import *

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import scipy.stats as stats

CMAP = matplotlib.cm.get_cmap('magma')

GRAY = "#808080"

def rgb2hex(r,g,b):
    r *= 255
    g *= 255
    b *= 255
    r = int(r)
    g = int(g)
    b = int(b)
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def init_data():
    support_data = np.load(os.path.join('/home/matthew/physics_atv_ws/src/learning/context_adaptation/assets', 'gp_params', 'support_data.npy'))

    from sklearn.manifold import TSNE

    all_data = support_data

    all_tsne = TSNE(n_components=2, perplexity=50, random_state=0).fit_transform(all_data[:,:8])

    plt.scatter(all_tsne[:,0], all_tsne[:,1])
    plt.show()

    print(all_tsne.shape)

    print(np.min(all_tsne, axis=0), np.max(all_tsne, axis=0))
    print(np.min(all_data[:,-2]), np.max(all_data[:,-2]))

    all_tsne[:,1] /= 25
    all_tsne[:,0] /= 17
    all_data[:,-2]/=4
    all_data[:,-2] -= 1

    # print(all_data[:,-2])

    return all_tsne, all_data

class NewScatter(Scene):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "camera_config": {"background_color":WHITE}
    }
    def construct(self):
        self.camera.background_color = WHITE
        scale = 3

        axis_range = scale
        axis_length = scale  # Total length for each axis

        # Setting up the axes with defined range and length
        axes = ThreeDAxes(
            x_range=[-axis_range, axis_range, 1],
            y_range=[-axis_range, axis_range, 1],
            z_range=[-axis_range, axis_range, 1],
            x_length=axis_length,
            y_length=axis_length,
            z_length=axis_length,
        )

        # bounding_box = VCube(side_length=axis_range + 5, color=GRAY, fill_opacity=0.01)
        # bounding_box.set_opacity(0.1)
        # bounding_box.set_fill(GRAY, opacity=0.1)
        # bounding_box.set_stroke(width=1, opacity=0.5)
        # bounding_box.move_to(ORIGIN)

        # self.camera.frame.scale(0.7)

        # Generate initial random points
        self.all_tsne, self.all_data = init_data()
        self.all_tsne *= scale
        self.all_data[:, -2] *= scale

        print(self.all_data[:, -2].min(), self.all_data[:, -2] .max())

        buff_size = 100
        buff_intelligent = self.all_data[:buff_size].copy()
        intelligent_tsne = self.all_tsne[:buff_size].copy()
        buff_classes = np.argmin(buff_intelligent[:, :8], axis=1)

        self.buff_FIFO = self.all_data[:buff_size].copy()
        self.FIFO_tsne = self.all_tsne[:buff_size].copy()
        self.buff_toi = np.arange(self.buff_FIFO.shape[0])

        self.buff_intelligent = self.all_data[:buff_size].copy()
        self.intelligent_tsne = self.all_tsne[:buff_size].copy()
        self.buff_classes = np.argmin(self.buff_intelligent[:,:8],axis=1)

        self.i = buff_size

        points = np.hstack([self.FIFO_tsne[:, :2], self.buff_FIFO[:, -2].reshape(-1, 1)])

        # print(points)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2])  # , c=buff_FIFO[:,-1])
        # ax.view_init(elev=15, azim=-64)
        # plt.show()

        point_colors = []
        for r in self.all_data[:, -1]:
            point_colors.append(rgb2hex(*CMAP(r)[:3]))

        # Create dots at the random points
        self.spheres = [Cube(side_length=.2, color=GRAY, fill_opacity=.8).move_to(point).set_color(color) for point, color in zip(points, point_colors)]

        # Add axes and spheres to the scene
        # self.add(axes, bounding_box)
        self.add(axes)
        for sphere in self.spheres:
            self.add(sphere)

        # Set the initial camera position
        # self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.camera.frame.set_euler_angles(
            phi=75 * DEGREES, theta=-45 * DEGREES
        )
        # self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES)

        # Rotate the scene in yaw
        # self.camera.frame.add_ambient_rotation(angular_speed=1 * DEGREES)  # Adjust the rate for faster/slower rotation

        # Create an updater function to replace the oldest point
        def update_dots(mob, dt):
            sample = self.all_data[self.i]

            # # FIFO UPDATE
            idx = np.argmin(self.buff_toi)
            self.buff_FIFO[idx] = sample
            self.buff_toi[idx] = self.i
            self.FIFO_tsne[idx] = self.all_tsne[self.i]

            #Intelligent UPDATE
            # most_class = stats.mode(self.buff_classes)[0]
            # vel_hist = torch.histc(torch.from_numpy(self.buff_intelligent)[self.buff_classes == most_class,-2], bins=10, min=-3,max=4.5)
            # most_vel = torch.argmax(vel_hist)
            # edges = np.arange(0,11).astype(float)
            # edges /= 4
            # edges -= 1
            # edges *= scale
            # # print(edges)
            # # print(torch.from_numpy(self.buff_intelligent)[self.buff_classes == most_class,-2])
            # sel_min = edges[most_vel]
            # sel_max = edges[most_vel+1]
            # train_vels = self.buff_intelligent[:,-2]
            # # print(sel_min, sel_max)
            # # print(train_vels)
            # insert_idx = np.random.choice(np.where((self.buff_classes == most_class) & (train_vels > sel_min) & (train_vels < sel_max))[0])
            # self.buff_intelligent[insert_idx] = sample
            # self.buff_classes[insert_idx] = np.argmin(sample[:8])
            # self.intelligent_tsne[insert_idx] = self.all_tsne[self.i]
            # idx = insert_idx
            # # print(idx)

            # Remove the oldest dot
            self.remove(self.spheres[idx])
            # Add a new dot at a new random point
            color = rgb2hex(*CMAP(sample[-1])[:3])
            new_sphere = Cube(side_length=.2, color=GRAY, fill_opacity=.8).move_to([self.FIFO_tsne[idx, 0], self.FIFO_tsne[idx, 1], self.buff_FIFO[idx, -2]]).set_color(color)
            # new_sphere = Cube(side_length=.2, color=GRAY, fill_opacity=.8).move_to([self.intelligent_tsne[idx, 0], self.intelligent_tsne[idx, 1], self.buff_intelligent[idx, -2]]).set_color(color)
                    # self.spheres = [Sphere(radius=0.08).move_to(point).set_color(color) for point, color in zip(points, point_colors)]

            self.spheres[idx] = new_sphere
            self.add(new_sphere)

            self.i += 1
            if self.i > 530:
                self.i = 0

        # Add the updater to the dots group
        self.camera.frame.add_updater(update_dots)
        self.camera.frame.add_updater(lambda m, dt: m.increment_theta(1 * DEGREES))

        # Let the scene run for a while to see the rotation and point replacement
        self.wait(8)
        self.camera.frame.remove_updater(update_dots)
        self.wait(8)

# if __name__ == "__main__":
#     scene = NewScatter()
#     scene.render()
