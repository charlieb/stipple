from constraints import radius_exclude, rect_limit
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from numba import jit, float64, int64
from random import random
import svgwrite as svg
import time
from math import sqrt
import sys

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
        if 1.00-(image[x][y] / 255.) > random():
            points[ip][0] = x
            points[ip][1] = y
            points[ip][2] = 1.
            #print(ip, points[ip])
            ip += 1

    total_err = 9999999
    while total_err > npoints / 10: # 1 pixel of error per point's not bad
        kdtree(points)
        pold = points
        # rebind with new array so KDTree will continue to work ok
        points = np.zeros((npoints,3), dtype='float64')
        
        # build coord/density list
        # i.e. sum([px,py]*(255 - pixel grey))
        for x,y in product(range(image.shape[0]), range(image.shape[1])):
            p = query([x,y,1.])
            d = 255.001-image[x][y]
            points[p] += [x*d, y*d, d]

        for p,po in zip(points,pold): 
            if p[2] > 0: 
                p /= p[2]
            else:
                np.copyto(p, po)

        #draw2(npoints, pold, points, image.shape[0], image.shape[1])
        #draw1(npoints, points, image.shape[0], image.shape[1])
                
        total_err = 0.
        for p1, p2 in zip(pold, points):
            er = p1-p2
            er = er**2
            total_err += sqrt(er[0]+er[1])
        print(total_err)

    # one last pass to give information about the average color of the region
    counts = np.zeros(npoints, dtype='int64')
    for x,y in product(range(image.shape[0]), range(image.shape[1])):
        p = query([x,y,1.])
        points[p][2] += 255-image[x][y]
        counts[p] += 1
    for p,c in zip(points, counts):
        if c > 0: p[2] /= c
        else: p[2] = 0.

    return points

def stipple(image, npoints, resize=400):
    im = Image.open(image).convert('L')
    w,h = im.size
    if w > h:
        nw = resize
        nh = int(resize * h/w)
    else:
        nw = int(resize * w/h)
        nh = resize
    im = im.resize((nw,nh))
    im = im.filter(ImageFilter.GaussianBlur(radius=1))

    np_im = np.array(im, dtype='float64')
    points = voronoi(npoints, np_im)
    return points

def draw(points, filename, mindot=0.75, maxdot=4.):
    minx = miny =  9999999
    maxx = maxy = -9999999
    dwg = svg.Drawing(filename)
    for p in points:
        minx = p[1] if p[1] < minx else minx
        maxx = p[1] if p[1] > maxx else maxx
        miny = p[0] if p[0] < miny else miny
        maxy = p[0] if p[0] > maxy else maxy
        # coordinate systems are reversed for image vs svg
        c = svg.shapes.Circle((p[1], p[0]), maxdot - (p[2] / 255.) * (maxdot - mindot),
                                    fill='none', 
                                    stroke='black',
                                    stroke_width=2.)
        dwg.add(c)
    dwg.viewbox(minx=minx, miny=miny, width=maxx-minx, height=maxy-miny)
    dwg.save()

def main():
    #im = Image.open('StipplingOriginals/plant2_400x400.png').convert('L')
    #im = Image.open('StippleGen2/data/grace.jpg').convert('L')
    if len(sys.argv) < 2:
        print("Usage: python stipple_lib.py <filename>")
        return
    points = stipple(sys.argv[1], 1000, resize=200)
    draw(points, 'vor.svg', mindot=0.275, maxdot=2.)


if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print(t1-t0)
