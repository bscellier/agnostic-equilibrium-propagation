import numpy as np
from PIL import ImageTk
from random import randrange
import threading
import tkinter
from tkinter import Button, Canvas, Entry, Label, Scale, StringVar, Tk
import torch
from training.epoch import Evaluator
from training.statistics import NumIterationsStat, ErrorFinder
import gui.utils
from datasets import MyDataset


class GUI(Tk):

    def __init__(self, network, dataset):

        Tk.__init__(self, None)

        self.title('Network')

        self._network = network
        self.dataset = dataset

        layer_shapes = network.layer_shapes()
        num_layers = len(layer_shapes)

        self.layer_shapes_4d  = [gui.utils.to_4d(shape) for shape in layer_shapes]
        self.layer_shapes_2d  = [gui.utils.to_2d(shape) for shape in layer_shapes]
        self.image_shapes = [gui.utils.to_shape_image(shape) for shape in self.layer_shapes_2d]


        y_coordinates = list(np.cumsum([dims[1]+2 for dims in self.image_shapes[::-1]]))[::-1]
        width = max([dims[0] for dims in self.image_shapes])
        indent = 200
        
        self.canvas = Canvas(self, width = indent+width, height = y_coordinates[0])
        self.canvas.pack(side=tkinter.BOTTOM)

        # FIELD TO INPUT THE INDEX OF THE SAMPLE (IN THE DATASET)
        Label(self, text="Image Index").pack(side=tkinter.LEFT)
        self.index = StringVar()
        self.index.set("0")
        Entry(self, textvariable=self.index, width=5).pack(side=tkinter.LEFT)

        self.label_switch = True

        def activate_label():
            self.label_switch = not self.label_switch
            self.label["text"] = "Display Label"

        self.label = Button(self, text="Display Label", command=activate_label)
        self.label.pack(side=tkinter.LEFT)


        self.idx_list = None
        self.computed = False

        # pb1 = Progressbar(ws, orient=tkinter.HORIZONTAL, length=100, mode='indeterminate')
        # pb1.pack(expand=True)


        def get_index():
            if not self.computed:

                def work():
                    b1["state"] = "disabled"
                    b1["text"] = "Computing"
                    dataloader = torch.utils.data.DataLoader(MyDataset(dataset), batch_size=512, shuffle=False)
                    evaluator = Evaluator(self._network, dataloader)
                    evaluator.add_statistic(NumIterationsStat())
                    stat = ErrorFinder(self._network)
                    evaluator.add_statistic(stat)
                    evaluator.run()
                    self.idx_list = stat.get()
                    b1["state"] = "normal"
                    b1["text"] = "Show a Misclassified Image"
                
                # t1 = threading.Thread(target=work)  # TODO: start a new thread to find the misclassified images while the GUI is still usable
                # t1.start()  # To do that, the network used to find the misclassified images should be a copy (not the same) of the network used by the GUI
                work()
                self.computed = True
            elif self.idx_list is not None:
                num_mistakes = len(self.idx_list)
                idx = randrange(num_mistakes)
                index = self.idx_list[idx]
                self.index.set(str(index))

        b1 = Button(self, text="Find Misclassified Images", command=get_index)
        b1.pack(side=tkinter.LEFT)

        def clear_state():
            network.init_layers()

        Button(self, text="Clear State", command=clear_state).pack(side=tkinter.LEFT)

        Label(self, text="Speed").pack(side=tkinter.LEFT)
        step_size_var = tkinter.DoubleVar()
        self.step_size = Scale(self, variable=step_size_var, from_=0.1, to=1., resolution=0.1, orient=tkinter.HORIZONTAL)
        self.step_size.set(0.1)
        self.step_size.pack(side=tkinter.LEFT)

        Label(self, text="Nudging Factor").pack(side=tkinter.LEFT)
        nudging_var = tkinter.DoubleVar()
        self.nudging = Scale(self, variable=nudging_var, from_=-0.5, to=0.5, resolution=0.1, orient=tkinter.HORIZONTAL)
        self.nudging.set(0.)
        self.nudging.pack(side=tkinter.LEFT)

        x, y = self.get_data()
        network.set_data(x, y)
        network.init_layers()

        self.imgTks = self.get_imgTks(x)
        self.img_canvas    = [self.canvas.create_image(indent + width//2, y_coord, anchor=tkinter.S, image=imgTk) for k, (imgTk, y_coord) in enumerate(zip(self.imgTks, y_coordinates))]
        
        energy = self._network.energy_fn().sum().item()
        self.energy_canvas = self.canvas.create_text( 20, 100, anchor=tkinter.W, font="Purisa", text="Energy = {:.1f}".format(energy))
        # self.cost_canvas   = self.canvas.create_text( 20, 200, anchor=tkinter.W, font="Purisa", text="Cost = %.4f"   % (cost))

        self.render()


    def get_data(self):

        index_as_string = self.index.get()  # get the index of the sample as a string that the user inputs

        # convert the string to an integer, or use the hash of the string if it is not a valid integer
        index = int(index_as_string) if index_as_string.isdigit() else hash(index_as_string)

        dataset_size = len(self.dataset)
        index = (index % dataset_size)  # forces the index to be a valid index of the dataset

        self.index.set(str(index))  # displays the index value in the entry (useful if the original string was not a valid integer)

        x, y = self.dataset[index]

        if self.label_switch:
            self.label["text"] = str(y)

        x = x.unsqueeze(0)  # add a dimension of size 1 as mini-batch size
        y = torch.as_tensor([y], dtype = torch.int64)

        return x, y


    def get_imgTks(self, x):

        all_layers = [x] + self._network.layers_state

        arrays = [256. * layer.detach().numpy().reshape(dims_4d, order='C').transpose(0,2,1,3).reshape(dims_2d, order='C') for layer, dims_4d, dims_2d in zip(all_layers, self.layer_shapes_4d, self.layer_shapes_2d)]
        images = [gui.utils.aux(array).resize(image_shape) for array, image_shape in zip(arrays, self.image_shapes)]
        imgTks = [ImageTk.PhotoImage(image) for image in images]

        return imgTks

    def render(self):

        x, y = self.get_data()
        self._network.set_data(x, y)

        nudging = self.nudging.get()
        self._network.set_nudging(nudging)

        step_size = self.step_size.get()
        self._network.gradient_step(step_size)

        self.imgTks = self.get_imgTks(x)
        for img_canvas, imgTk in zip(self.img_canvas, self.imgTks):
            self.canvas.itemconfig(img_canvas, image=imgTk)

        energy = self._network.energy_fn().sum().item()
        self.canvas.itemconfig(self.energy_canvas, text="Energy = {:.1f}".format(energy))

        self.canvas.after(100, self.render)
