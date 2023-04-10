import pygame
import math
import numpy as np
import numba as nb
from PIL import Image


pygame.init()

screen = pygame.display.set_mode((500,300))

screen_x, screen_y = screen.get_size()


@nb.njit(fastmath=True)
def ray_triangle_intersection(
    vertices: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    epsilon: float = 1e-6
):
    vertex_0 = vertices[0]
    vertex_1 = vertices[1]
    vertex_2 = vertices[2]

    edge_1 = vertex_1 - vertex_0
    edge_2 = vertex_2 - vertex_0

    p_vec = np.cross(ray_direction, edge_2)

    determinant = np.dot(p_vec, edge_1)

    if np.abs(determinant) < epsilon:
        return False, 0.0, 0.0, 0.0

    inv_determinant = 1.0 / determinant

    t_vec = ray_origin - vertex_0
    u = np.dot(p_vec, t_vec) * inv_determinant

    if u < 0.0 or u > 1.0:
        return False, 0.0, 0.0, 0.0

    q_vec = np.cross(t_vec, edge_1)
    v = np.dot(q_vec, ray_direction) * inv_determinant

    if v < 0.0 or (u + v) > 1.0:
        return False, 0.0, 0.0, 0.0

    t = np.dot(q_vec, edge_2) * inv_determinant

    if t < epsilon:
        return False, 0.0, 0.0, 0.0

    return True, t, u, v


@nb.njit(fastmath=True)
def rotate_points(points, origin, rotation_matrix):
    rotated_points = []

    for point in points:
        rotated_points.append(np.reshape(np.dot(rotation_matrix, np.reshape(point - origin, (3,1))) + np.reshape(origin, (3,1)), (1,3))[0])

    return rotated_points


@nb.njit(fastmath=True)
def texture_map(texture, intersection_point, center, size, rotation_matrix):
    point = np.reshape(np.dot(np.linalg.inv(rotation_matrix), np.reshape(intersection_point - center, (3,1))) + np.reshape(center, (3,1)), (1,3))[0] - center

    abs_x = abs(point[0])
    abs_y = abs(point[1])
    abs_z = abs(point[2])

    coord = max(abs_x, abs_y, abs_z)

    if coord == point[0]: # right
        texture_pos = (point[2]+size, point[1]+size)
    elif coord == -point[0]: # left
        texture_pos = (size-point[2], point[1]+size)
    elif coord == point[1]: # top
        texture_pos = (point[0]+size, point[2]+size)
    elif coord == -point[1]: # bottom
        texture_pos = (point[0]+size, point[2]+size)
    elif coord == point[2]: # front
        texture_pos = (point[0]+size, point[1]+size)
    else: # back
        texture_pos = (size-point[0], point[1]+size)
    
    return texture[round(texture_pos[0])][round(texture_pos[1])]


class Triangle:
    def __init__(self, p1, p2, p3, color=None, texture=None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.vertices = np.array([p1, p2, p3], dtype=np.float64)
        self.color = color
        self.texture = texture


class Cube:
    def __init__(self, center, full_size, x_rot=0, y_rot=0, z_rot=0, color=(255,255,255), texture=None):
        self.center = center
        size = full_size/2
        self.size = size
        self.x_rot = x_rot
        self.y_rot = y_rot
        self.z_rot = z_rot
        self.color = color
        self.texture = texture

        if self.texture:
            self.texture = np.array(self.texture.resize((full_size, full_size)))

        _z_rot_cos = math.cos(z_rot)
        _z_rot_sin = math.sin(z_rot)
        _y_rot_cos = math.cos(y_rot)
        _y_rot_sin = math.sin(y_rot)
        _x_rot_cos = math.cos(x_rot)
        _x_rot_sin = math.sin(x_rot)

        z_rotation_matrix = np.array([
            [_z_rot_cos, -_z_rot_sin, 0],
            [_z_rot_sin, _z_rot_cos, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        y_rotation_matrix = np.array([
            [_y_rot_cos, 0, _y_rot_sin],
            [0, 1, 0],
            [-_y_rot_sin, 0, _y_rot_cos]
        ], dtype=np.float64)
        x_rotation_matrix = np.array([
            [1, 0, 0],
            [0, _x_rot_cos, -_x_rot_sin],
            [0, _x_rot_sin, _x_rot_cos]
        ], dtype=np.float64)

        rotation_matrix = np.dot(np.dot(z_rotation_matrix, y_rotation_matrix), x_rotation_matrix)
        self.rotation_matrix = rotation_matrix

        front_top_left = rotate_points(np.array([[center[0]-size, center[1]+size, center[2]-size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        front_top_right = rotate_points(np.array([[center[0]+size, center[1]+size, center[2]-size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        front_bottom_left = rotate_points(np.array([[center[0]-size, center[1]-size, center[2]-size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        front_bottom_right = rotate_points(np.array([[center[0]+size, center[1]-size, center[2]-size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        back_top_left = rotate_points(np.array([[center[0]-size, center[1]+size, center[2]+size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        back_top_right = rotate_points(np.array([[center[0]+size, center[1]+size, center[2]+size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        back_bottom_left = rotate_points(np.array([[center[0]-size, center[1]-size, center[2]+size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        back_bottom_right = rotate_points(np.array([[center[0]+size, center[1]-size, center[2]+size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        self.triangles = [
            Triangle(front_top_left, back_top_left, back_top_right, color=color),
            Triangle(front_top_left, front_top_right, back_top_right, color=color),
            Triangle(front_top_left, back_top_left, back_bottom_left, color=color),
            Triangle(front_top_left, front_bottom_left, back_bottom_left, color=color),
            Triangle(front_top_left, front_bottom_left, front_bottom_right, color=color),
            Triangle(front_top_left, front_top_right, front_bottom_right, color=color),
            Triangle(back_bottom_right, front_bottom_right, back_top_right, color=color),
            Triangle(back_bottom_right, front_bottom_right, front_bottom_left, color=color),
            Triangle(back_bottom_right, back_bottom_left, front_bottom_left, color=color),
            Triangle(back_bottom_right, back_bottom_left, back_top_left, color=color),
            Triangle(back_bottom_right, back_top_right, back_top_left, color=color)
        ]
    
    def rotate(self, x_rot=0, y_rot=0, z_rot=0):
        self.x_rot += x_rot
        x_rot = self.x_rot
        self.y_rot += y_rot
        y_rot = self.y_rot
        self.z_rot += z_rot
        z_rot = self.z_rot

        _z_rot_cos = math.cos(z_rot)
        _z_rot_sin = math.sin(z_rot)
        _y_rot_cos = math.cos(y_rot)
        _y_rot_sin = math.sin(y_rot)
        _x_rot_cos = math.cos(x_rot)
        _x_rot_sin = math.sin(x_rot)

        z_rotation_matrix = np.array([
            [_z_rot_cos, -_z_rot_sin, 0],
            [_z_rot_sin, _z_rot_cos, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        y_rotation_matrix = np.array([
            [_y_rot_cos, 0, _y_rot_sin],
            [0, 1, 0],
            [-_y_rot_sin, 0, _y_rot_cos]
        ], dtype=np.float64)
        x_rotation_matrix = np.array([
            [1, 0, 0],
            [0, _x_rot_cos, -_x_rot_sin],
            [0, _x_rot_sin, _x_rot_cos]
        ], dtype=np.float64)

        rotation_matrix = np.dot(np.dot(z_rotation_matrix, y_rotation_matrix), x_rotation_matrix)
        self.rotation_matrix = rotation_matrix

        size = self.size
        center = self.center
        color = self.color

        front_top_left = rotate_points(np.array([[center[0]-size, center[1]+size, center[2]-size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        front_top_right = rotate_points(np.array([[center[0]+size, center[1]+size, center[2]-size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        front_bottom_left = rotate_points(np.array([[center[0]-size, center[1]-size, center[2]-size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        front_bottom_right = rotate_points(np.array([[center[0]+size, center[1]-size, center[2]-size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        back_top_left = rotate_points(np.array([[center[0]-size, center[1]+size, center[2]+size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        back_top_right = rotate_points(np.array([[center[0]+size, center[1]+size, center[2]+size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        back_bottom_left = rotate_points(np.array([[center[0]-size, center[1]-size, center[2]+size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        back_bottom_right = rotate_points(np.array([[center[0]+size, center[1]-size, center[2]+size]]), np.array(center, dtype=np.float64), rotation_matrix)[0]
        self.triangles = [
            Triangle(front_top_left, back_top_left, back_top_right, color=color),
            Triangle(front_top_left, front_top_right, back_top_right, color=color),
            Triangle(front_top_left, back_top_left, back_bottom_left, color=color),
            Triangle(front_top_left, front_bottom_left, back_bottom_left, color=color),
            Triangle(front_top_left, front_bottom_left, front_bottom_right, color=color),
            Triangle(front_top_left, front_top_right, front_bottom_right, color=color),
            Triangle(back_bottom_right, front_bottom_right, back_top_right, color=color),
            Triangle(back_bottom_right, front_bottom_right, front_bottom_left, color=color),
            Triangle(back_bottom_right, back_bottom_left, front_bottom_left, color=color),
            Triangle(back_bottom_right, back_bottom_left, back_top_left, color=color),
            Triangle(back_bottom_right, back_top_right, back_top_left, color=color)
        ]


class Ray:
    def __init__(self, origin, direction, screen_pixel_pos):
        self.origin = origin
        self.direction = direction
        self.screen_pixel_pos = screen_pixel_pos


class Camera:
    def __init__(self, pos, render_distance=100):
        self.pos = np.array(pos, dtype=np.float64)
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.render_distance = render_distance
        self.rays = []

        for y in range(screen_y):
            for x in range(screen_x):
                self.rays.append(Ray(self.pos, np.array([
                    self.x + x - screen_x / 2,
                    self.y - y + screen_y / 2,
                    self.z + self.render_distance
                ], dtype=np.float64), (x, y)))
    
    def move_x(self, dx):
        self.x += dx
        self.pos[0] = self.x

        for ray in self.rays:
            ray.origin[0] += dx
            ray.direction[0] += dx

    def move_y(self, dy):
        self.y += dy
        self.pos[1] = self.y

        for ray in self.rays:
            ray.origin[1] += dy
            ray.direction[1] += dy
    
    def move_z(self, dz):
        self.z += dz
        self.pos[2] = self.z

        for ray in self.rays:
            ray.origin[2] += dz
            ray.direction[2] += dz
    
    def rotate(self, y_rot, x_rot):
        _y_rot_cos = math.cos(y_rot)
        _y_rot_sin = math.sin(y_rot)
        _x_rot_cos = math.cos(x_rot)
        _x_rot_sin = math.sin(x_rot)

        y_rotation_matrix = np.array([
            [_y_rot_cos, 0, _y_rot_sin],
            [0, 1, 0],
            [-_y_rot_sin, 0, _y_rot_cos]
        ], dtype=np.float64)
        x_rotation_matrix = np.array([
            [1, 0, 0],
            [0, _x_rot_cos, -_x_rot_sin],
            [0, _x_rot_sin, _x_rot_cos]
        ], dtype=np.float64)

        rotation_matrix = np.dot(y_rotation_matrix, x_rotation_matrix)

        for i, rotated_ray in enumerate(rotate_points(np.array([ray.direction for ray in self.rays]), self.pos, rotation_matrix)):
            self.rays[i].direction = rotated_ray


class Scene:
    def __init__(self, objects=[], camera=Camera((0,0,0))):
        self.objects = objects
        self.camera = camera
    
    def render(self, screen):
        for ray in self.camera.rays:
            for obj in self.objects:
                intersections = []

                for triangle in obj.triangles:
                    intersection = ray_triangle_intersection(triangle.vertices, ray.origin, ray.direction)

                    if intersection[0]:
                        intersections.append(np.array(intersection[1:], dtype=np.float64))
                if len(intersections) != 0:
                    if obj.texture is not None:
                        closest = math.inf
                        intersection_point = None

                        for intersection in intersections:
                            dist = np.linalg.norm(self.camera.pos - intersection)

                            if dist < closest:
                                closest = dist
                                intersection_point = intersection

                        screen.set_at(ray.screen_pixel_pos, texture_map(obj.texture, intersection_point, np.array(obj.center, dtype=np.float64), obj.size, obj.rotation_matrix).tolist()[:-1])
                    else:
                        screen.set_at(ray.screen_pixel_pos, triangle.color)


def main():
    clock = pygame.time.Clock()

    scene = Scene(objects=[Cube((0,0,50), 40, texture=Image.open("Hu Tao.png"))])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        '''keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            scene.camera.move_z(1)
        if keys[pygame.K_s]:
            scene.camera.move_z(-1)
        if keys[pygame.K_a]:
            scene.camera.move_x(-1)
        if keys[pygame.K_d]:
            scene.camera.move_x(1)
        if keys[pygame.K_UP]:
            scene.camera.move_y(1)
        if keys[pygame.K_DOWN]:
            scene.camera.move_y(-1)'''

        #mouse_pos = pygame.mouse.get_pos()
        #dx, dy = (mouse_pos[0] - screen_x/2)/screen_x, (mouse_pos[1] - screen_y/2)/screen_y
        #scene.camera.rotate(dx, dy)

        scene.objects[0].rotate(x_rot=0.1,y_rot=0.1,z_rot=0.1)

        screen.fill("black")

        scene.render(screen)

        pygame.display.flip()
        clock.tick(30)
        #print("tick")

if __name__ == "__main__":
    main()