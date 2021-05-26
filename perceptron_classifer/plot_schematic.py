'''Plot schematic illustrating forward pass.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2021-05-17 14:00:46.
'''

from __future__ import print_function
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




#-------------Main---------------------------------
if __name__=='__main__':

    g = nx.DiGraph()

    # nodes: (l, i), use l as x-coor, i as y-coor
    x0 = 0
    xl = 2
    xl1 = 5
    layer0 = [(x0, 0), (x0, 2), (x0, 7), (x0, 9)]
    layeri = [(xl, 0), (xl, 5), (xl, 9)]
    layeri1 = [(xl1, 0), (xl1, 5), (xl1, 9)]

    layers = [layer0, layeri, layeri1]
    pos = {}

    # add nodes
    for lii in layers:
        g.add_nodes_from(lii)
        pos.update(dict(zip(lii, lii)))

    # add bias
    bias = (3.5, 11)
    g.add_node(bias)
    pos[bias] = bias

    # add links
    g.add_edge(layeri[1], layeri1[1])
    g.add_edge(bias, layeri1[1])

    # plot nodes
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    nx.draw(g, pos=pos, ax=ax, node_size=2800, with_labels=False,
            node_color='c', alpha=0.7)

    import matplotlib.patches as patches
    rect = patches.Rectangle((x0-0.5, -0.9), 1.0, 10.9, facecolor='y', alpha=0.2)
    ax.add_patch(rect)
    rect = patches.Rectangle((xl-0.5, -0.9), 1.0, 10.9, facecolor='y', alpha=0.2)
    ax.add_patch(rect)
    rect = patches.Rectangle((xl1-0.5, -0.9), 1.0, 10.9, facecolor='y', alpha=0.2)
    ax.add_patch(rect)

    # add activation
    ax.arrow(xl1+0.2, 5, 1, 0, head_width=0.05, head_length=0.1)

    # add ellipses
    fontsize=28
    ax.text(x0, 4.5, r'$\vdots$', fontsize=fontsize, va='center', ha='center')
    ax.text(xl, 2.5, r'$\vdots$', fontsize=fontsize, va='center', ha='center')
    ax.text(xl, 7, r'$\vdots$', fontsize=fontsize, va='center', ha='center')
    ax.text(xl1, 2.5, r'$\vdots$', fontsize=fontsize, va='center', ha='center')
    ax.text(xl1, 7, r'$\vdots$', fontsize=fontsize, va='center', ha='center')

    # label layers
    fontsize=14
    ax.text(x0, -1.2, 'l = 0', fontsize=fontsize, va='center', ha='center', fontname='serif')
    ax.text(xl, -1.2, 'l = l', fontsize=fontsize, va='center', ha='center', fontname='serif')
    ax.text(xl1, -1.2, 'l = l+1', fontsize=fontsize, va='center', ha='center', fontname='serif')

    # label neurons
    fontsize=16
    ax.text(xl, 5, 'i', fontsize=fontsize, va='center', ha='center')
    ax.text(xl1, 5, 'j', fontsize=fontsize, va='center', ha='center')
    ax.text(bias[0], bias[1], 'bias', fontsize=fontsize, va='center', ha='center')
    ax.text(xl, 9, r'$0$', fontsize=fontsize, va='center', ha='center')
    ax.text(xl1, 9, r'$0$', fontsize=fontsize, va='center', ha='center')
    ax.text(xl, 0, r'$s_{l}-1$', fontsize=fontsize-3, va='center', ha='center')
    ax.text(xl1, 0, r'$s_{l+1}-1$', fontsize=fontsize-3, va='center', ha='center')

    # equation
    fontsize=16
    ax.text(0.5*(xl + xl1), 5.3, r'$ a_{i}^{(l)} \cdot \theta_{j,i}^{(l+1)}$',
            fontsize=fontsize, va='bottom', ha='center', bbox=dict(facecolor='g', alpha=0.3))
    ax.text(0.5*(bias[0] + xl1)-0.6, 0.5*(bias[1] + 5),
            r'$b_{j}^{(l+1)}$',
            fontsize=fontsize, va='bottom', ha='center', bbox=dict(facecolor='g', alpha=0.3))
    ax.text(xl1+1.4, 5.3, r'$a_{j}^{(l+1)} = g(z_{j}^{(l+1)})$',
            fontsize=fontsize, va='bottom', ha='center', bbox=dict(facecolor='g', alpha=0.3))


    ax.set_xlim(-2, 7)
    ax.set_ylim(-3, 12)


    plot_save_name='schematic_nn_forward.png'
    print('\n# <plot_schematic>: Save figure to', plot_save_name)
    fig.savefig(plot_save_name,dpi=100,bbox_inches='tight')

    fig.show()
