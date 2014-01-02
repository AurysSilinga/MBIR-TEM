# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:06:42 2014

@author: Jan
"""

import vtk

reader = vtk.vtkDataSetReader()

reader.SetFileName('test.vtk')

reader.ReadAllScalarsOn()

reader.Update()

output = reader.GetOutput()
scalar_range = output.GetScalarRange()
 
# Create the mapper that corresponds the objects of the vtk file into graphics elements
mapper = vtk.vtkDataSetMapper()
mapper.SetInput(output)
mapper.SetScalarRange(scalar_range)
 
# Create the Actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
 
# Create the Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1, 1, 1) # Set background to white
 
# Create the RendererWindow
renderer_window = vtk.vtkRenderWindow()
renderer_window.AddRenderer(renderer)
 
# Create the RendererWindowInteractor and display the vtk_file
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderer_window)
interactor.Initialize()
interactor.Start()





print reader.GetHeader()

print 'Points:', output.GetNumberOfPoints()
print 'Cells: ', output.GetNumberOfCells()
#print 'Polys: ', output.GetNumberOfPolys()
#print 'Lines: ', output.GetNumberOfLines()
#print 'Strips:', output.GetNumberOfStrips()
#print 'Piece: ', output.GetNumberOfPiece()
#print 'Verts: ', output.GetNumberOfVerts()

points = output.GetPoints()

print output.GetPoint(0)
