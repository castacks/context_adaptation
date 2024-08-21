from manim import *
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
CMAP = matplotlib.cm.get_cmap('magma')

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

    all_tsne /= 25
    all_data[:,-2]/=4
    all_data[:,-2] -= 1

    # print(all_data[:,-2])

    return all_tsne, all_data

from manim import *




class NewScatter(ThreeDScene):
    def construct(self):
        self.camera.background_color = WHITE
        # Setting up the axes
        # axes = ThreeDAxes()

        # Define the range for the axes
        # axis_range = 1
        #
        # # Setting up the axes with defined range
        # axes = ThreeDAxes(
        #     x_range=[-axis_range, axis_range, 1],
        #     y_range=[-axis_range, axis_range, 1],
        #     z_range=[-axis_range, axis_range, 1],
        # )
        scale = 5

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

        bounding_box = Cube(side_length=axis_range + 5, color=GRAY, fill_opacity=0.1)
        bounding_box.move_to(ORIGIN)

        self.camera.set_zoom(0.7)

        # Generate initial random points
        # num_points = 50
        # points = [self.random_point() for _ in range(num_points)]
        self.all_tsne, self.all_data = init_data()
        self.all_tsne *= scale
        self.all_data[:,-2] *= scale

        buff_size = 200
        t = buff_size
        buff_intelligent = self.all_data[:buff_size].copy()
        intelligent_tsne = self.all_tsne[:buff_size].copy()
        buff_classes = np.argmin(buff_intelligent[:,:8],axis=1)

        self.buff_FIFO = self.all_data[:buff_size].copy()
        self.FIFO_tsne = self.all_tsne[:buff_size].copy()
        self.buff_toi = np.arange(self.buff_FIFO.shape[0])

        self.i = buff_size
        # print(FIFO_tsne[:,:2].shape, buff_FIFO[:,-2].shape)

        points = np.hstack([self.FIFO_tsne[:,:2], self.buff_FIFO[:,-2].reshape(-1,1)])

        print(points)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2])#, c=buff_FIFO[:,-1])
        ax.view_init(elev=15,azim=-64)
        plt.show()


        point_colors = []
        for r in self.all_data[:,-1]:
            point_colors.append(rgb2hex(*CMAP(r)[:3]))

        # Create dots at the random points
        dots = VGroup(*[Dot3D(point, color=color) for point,color in zip(points,point_colors)])

        # Add axes and dots to the scene
        self.add(axes, dots, bounding_box)

        # Set the initial camera position
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # Rotate the scene in yaw
        self.begin_ambient_camera_rotation(rate=0.1)  # Adjust the rate for faster/slower rotation

        # Create an updater function to replace the oldest point
        def update_dots(mob, dt):
            sample = self.all_data[self.i]

            #FIFO UPDATE
            idx = np.argmin(self.buff_toi)
            self.buff_FIFO[idx] = sample
            self.buff_toi[idx] = self.i
            self.FIFO_tsne[idx] = self.all_tsne[self.i]

            # Remove the oldest dot
            mob.remove(mob[idx])
            # Add a new dot at a new random point
            color = rgb2hex(*CMAP(sample[-1])[:3])
            new_dot = Dot3D([self.FIFO_tsne[idx,0],self.FIFO_tsne[idx,1], self.buff_FIFO[idx,-2]], color=color)
            mob.add(new_dot)

            self.i += 1

        # Add the updater to the dots group
        dots.add_updater(update_dots)

        # Let the scene run for a while to see the rotation and point replacement
        self.wait(20)

    def random_point(self):
        """Generate a random 3D point within a specified range."""
        return np.array([np.random.uniform(-3, 3),
                         np.random.uniform(-3, 3),
                         np.random.uniform(-3, 3)])


if __name__ == "__main__":
    from manim import config
    config.media_width = "100%"
    config.quality = "high_quality"
    config.pixel_height = 1080
    config.pixel_width = 1080
    config.format = "gif"
    from manim import scene_classes
    scene_classes.main(Scatter3DScene)
