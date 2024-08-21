from manim import *

class ImageIn3DScene(ThreeDScene):
    def construct(self):
        # Create a 2D image
        image = ImageMobject("/home/matthew/Downloads/000014.png")
        image.scale(2)

        # Move the image to a 3D position
        image.move_to([1, 1, 1])

        # Add the 2D image to the 3D scene
        self.add(image)

        # Set up the 3D camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Add some 3D objects for context
        axes = ThreeDAxes()
        sphere = Sphere(radius=1, color=BLUE).move_to([1, 1, 2])

        self.add(axes, sphere)

        self.begin_ambient_camera_rotation(rate=0.1)

        self.wait(10)
