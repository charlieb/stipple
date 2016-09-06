from constraints import radius_exclude, rect_limit
from PIL import Image, ImageOps
import numpy as np
from numba import jit, float64, int64
from random import random
import svgwrite as svg
import time
from math import sqrt

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree
from itertools import product

def draw_voronoi(npoints, points, sizex, sizey):
    dwg = svg.Drawing('voronoi%4d.svg'%frame)
    cols = list(svg.data.colors.colornames)*npoints
    tree = KDTree(points)

    for x,y in product(range(sizex), range(sizey)):
        # coordinate systems are reversed for image vs svg
        _, region = tree.query([x,y, 1.])
        c = svg.shapes.Circle((x,y), 1,
                                    fill=cols[region],
                                    stroke='none',
                                    stroke_width=1.)
        dwg.add(c)
    dwg.viewbox(minx=0, miny=0, 
                width=sizex, height=sizey)
    dwg.save()

frame = 0
def draw2(npoints, points1, points2, sizex, sizey):
    global frame
    dwg = svg.Drawing('2voronoi%04d.svg'%frame)
    frame += 1
    for p1,p2 in zip(points1, points2):
        # coordinate systems are reversed for image vs svg
        c = svg.shapes.Circle((p1[1], p1[0]), 2,#1+np_im[p[0]][p[1]]/25, #p[2],
                                    fill='none', 
                                    stroke='black',
                                    stroke_width=0.1)
        dwg.add(c)
        c = svg.shapes.Circle((p2[1], p2[0]), 2,#1+np_im[p[0]][p[1]]/25, #p[2],
                                    fill='none', 
                                    stroke='red',
                                    stroke_width=0.1)
        dwg.add(c)
    dwg.viewbox(minx=0, miny=0, width=sizex, height=sizey)
    dwg.save()


draw1_frame = 0
def draw1(npoints, points, sizex, sizey):
    global draw1_frame
    fn = '1stipple%04d.svg'%draw1_frame
    print(fn)
    dwg = svg.Drawing(fn)
    draw1_frame += 1
    for p in points:
        # coordinate systems are reversed for image vs svg
        c = svg.shapes.Circle((p[1], p[0]), 1,#1+np_im[p[0]][p[1]]/25, #p[2],
                                    fill='black', 
                                    stroke='none',
                                    stroke_width=0.1)
        dwg.add(c)
    dwg.viewbox(minx=0, miny=0, width=sizex, height=sizey)
    dwg.save()


tree = None
def kdtree(points):
    global tree
    tree = KDTree(points)
    
def query(point):
    global tree
    p = tree.query(point)
    return p[1]

@jit(float64[:,:](int64, float64[:,:]))
def voronoi(npoints, image):
    points = np.zeros((npoints,3), dtype='float64')
    ip = 0
    while ip < npoints:
        x = int(random() * image.shape[0])
        y = int(random() * image.shape[1])
        if 1.001-(image[x][y] / 255.) > random():
            points[ip][0] = x
            points[ip][1] = y
            points[ip][2] = 1.
            #print(ip, points[ip])
            ip += 1

    total_err = 9999999
    while total_err > npoints / 1000: # 1 pixel of error per point's not bad
        kdtree(points)
        pold = points
        # rebind with new array so KDTree will continue to work ok
        points = np.zeros((npoints,3), dtype='float64')
        
        # build coord/density list
        # i.e. sum([px,py]*(255 - pixel grey))
        downsample = 2
        for xd,yd in product(range(int(image.shape[0] / downsample)), range(int(image.shape[1] / downsample))):
            x = xd * downsample
            y = yd * downsample
            p = query([x,y,1.])
            d = 255.001-image[x][y]
            points[p] += [x*d, y*d, d]
        for p,po in zip(points,pold): 
            if p[2] > 0: 
                p /= p[2]
            else:
                p = po

        #draw2(npoints, pold, points, image.shape[0], image.shape[1])
        draw1(npoints, points, image.shape[0], image.shape[1])
                
        total_err = 0.
        for p1, p2 in zip(pold, points):
            er = p1[:2]-p2[:2]
            er = er**2
            total_err += sqrt(er[0]+er[1])
        print(total_err)

    return points

@jit
def grey(p): return p[2]

@jit(float64[:,:](int64, int64[:,:]))
def add_points(npoints, image):
    points = np.zeros((npoints,3), dtype='float64')
    ip = 0
    while ip < npoints:
        x = int(random() * image.shape[0])
        y = int(random() * image.shape[1])
        if 1.-(image[x][y] / 255.) > random():
            points[ip][0] = x
            points[ip][1] = y
            points[ip][2] = 1.
            #print(ip, points[ip])
            ip += 1

    for _ in range(300):
        for i in range(npoints):
            p = points[i]
            #p[2] = max(1, image[int(p[0])][int(p[1])] / 64.)
            # shake it up a bit with a jitter
            #p[0] += sqrt(2) * (0.5 - random())
            #p[1] += sqrt(2) * (0.5 - random())
            
            # try to move towards the dark
            #x = int(p[0])
            #y = int(p[1])
            #dirs = [(x,y,image[x][y])]
            #if x > 0: dirs.append((x-1,y,image[x-1][y]))
            #if x < image.shape[0]:dirs.append((x+1,y,image[x+1][y]))
            #if y > 0: dirs.append((x,y-1,image[x][y-1]))
            #if y < image.shape[1]:dirs.append((x,y+1,image[x][y+1]))
            #pnew = min(dirs, key=grey)
            #p[0] = pnew[0]
            #p[1] = pnew[1]
            p[2] = max(1, image[int(p[0])][int(p[1])] / 64.)
            
        radius_exclude(npoints, points)
        rect_limit(0., 0., float(image.shape[0]), float(image.shape[1]), npoints, points)
        draw1(npoints, points, image.shape[0], image.shape[1])
    return points


def main():
    im = Image.open('StipplingOriginals/plant2_400x400.png').convert('L')
    im.show()
    np_im = np.array(im, dtype='float64')
    #np_im = np.zeros((128,128), dtype='float64')
    #for i in range(128):
    #    np_im[i].fill(i*2)

    #im = Image.fromarray(np_im)
    #im.show()
    npoints = 10000
    #points = add_points(npoints, np_im)
    points = voronoi(npoints, np_im)
    #draw_voronoi(npoints, points, np_im.shape[1], np_im.shape[0])

    dwg = svg.Drawing('vor.svg')
    for p in points:
        # coordinate systems are reversed for image vs svg
        c = svg.shapes.Circle((p[1]*10, p[0]*10), 1,# 1+np_im[p[0]][p[1]]/25, #p[2],
                                    fill='black', 
                                    stroke='none',
                                    stroke_width=1.)
        dwg.add(c)
    dwg.viewbox(minx=0, miny=0, 
                width=np_im.shape[1]*10, height=np_im.shape[0]*10)
    dwg.save()

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print(t1-t0)
