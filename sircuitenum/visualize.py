__doc__ = "visualize.py: Contains functions to visualize circuits and draw diagrams"
__author__ = "Eli Weissler, Mohit Bhat"
__version__ = "0.1.0"
__all__ = ["draw_circuit_diagram", "draw_basegraph", "draw_circuit_graph"]

from pathlib import Path
from typing import Union

import numpy as np
import networkx as nx
import matplotlib.transforms
import matplotlib
import matplotlib.pyplot as plt
import schemdraw.elements
from tqdm import tqdm
from scipy.spatial.distance import pdist
import schemdraw
import schemdraw.elements as elm

from sircuitenum import utils
from sircuitenum import reduction as red
from sircuitenum import visualize as viz

G_POS_BG = {2: [{0: np.array([-1/np.sqrt(2), -1/np.sqrt(2)]),
                1: np.array([0., 0.])}],
         3: [{0: np.array([0., 0.]),
              1: np.array([0.5, np.sqrt(3)/2]),
              2: np.array([-0.5, np.sqrt(3)/2])}]*2,
         4: [{0: np.array([0., 0.]),
                1: np.array([1., 1.])/np.sqrt(2),
                2: np.array([1., 0.])/np.sqrt(2),
                3: np.array([0., 1.])/np.sqrt(2)}]*6,
         5:   [{0: np.array([0, 1]),
                1: np.array([np.sin(2*2*np.pi/5), np.cos(2*2*np.pi/5)]),
                2: np.array([np.sin(4*2*np.pi/5), np.cos(4*2*np.pi/5)]),
                3: np.array([np.sin(1*2*np.pi/5), np.cos(1*2*np.pi/5)]),
                4: np.array([np.sin(3*2*np.pi/5), np.cos(3*2*np.pi/5)])}]*21}

G_POS = {2: [{0: np.array([-1/np.sqrt(2), -1/np.sqrt(2)]),
                1: np.array([0., 0.])}],
         3: [{0: np.array([0., 0.]),
              1: np.array([0.5, np.sqrt(3)/2]),
              2: np.array([-0.5, np.sqrt(3)/2])}]*2,
         4: [{0: np.array([np.sqrt(3)/2, -0.5]),
                1: np.array([0., 1.]),
                2: np.array([-np.sqrt(3)/2, -0.5]),
                3: np.array([0., 0.])},
               {0: -1*np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                1: np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                2: -2*np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                3: np.array([0., 0.])},
               {0: np.array([-0.5, np.sqrt(3)/2]),
                1: np.array([0., -1.]),
                2: np.array([0.5, np.sqrt(3)/2]),
                3: np.array([0., 0.])},
               {0: np.array([0., 0.]),
                1: np.array([1., -1.]),
                2: np.array([0., -1.]),
                3: np.array([1., 0.])},
               {0: np.array([-0.5, np.sqrt(3)/2]),
                1: np.array([1., 0.]),
                2: np.array([0.5, np.sqrt(3)/2]),
                3: np.array([0., 0.])},
               {0: (3/np.sqrt(3))*np.array([0.5, np.sqrt(3)/5]),
                1: (3/np.sqrt(3))*np.array([1., 0.]),
                2: (3/np.sqrt(3))*np.array([0.5, np.sqrt(3)/2]),
                3: (3/np.sqrt(3))*np.array([0., 0.])}],
         5:   [{0: np.array([-1, 0]),
                1: np.array([0, 1]),
                2: np.array([1, 0]),
                3: np.array([0, -1]),
                4: np.array([0, 0])},
               {0: np.array([0., -1.]),
                1: np.array([-0.5, np.sqrt(3)/2]),
                2: np.array([0.5, np.sqrt(3)/2]),
                3: np.array([0, -2]),
                4: np.array([0., 0.])},
               {0: np.array([-0.5, -np.sqrt(3)/2]),
                1: np.array([-0.5, np.sqrt(3)/2]),
                2: np.array([0.5, np.sqrt(3)/2]),
                3: np.array([0.5, -np.sqrt(3)/2]),
                4: np.array([0., 0.])},
               {0: np.array([-0.5, -np.sqrt(3)/2]),
                1: np.array([1, 0]),
                2: np.array([0, 1]),
                3: np.array([0.5, -np.sqrt(3)/2]),
                4: np.array([0., 0.])},
               {0: np.array([0, -1]),
                1: np.array([1, 0]),
                2: np.array([0, 1]),
                3: np.array([1, -1]),
                4: np.array([0., 0.])},
               {0: np.array([-0.5, -np.sqrt(3)/2]),
                1: np.array([1, 0]),
                2: np.array([0.5, np.sqrt(3)/2]),
                3: np.array([0.5, -np.sqrt(3)/2]),
                4: np.array([0., 0.])},
               {0: np.array([0, -1]),
                1: np.array([0.5, -0.5]),
                2: np.array([1, 0]),
                3: np.array([1, -1]),
                4: np.array([0., 0.])},
               {0: np.array([0, -1]),
                1: np.array([0.75, -0.25]),
                2: np.array([1, 0]),
                3: np.array([1, -1]),
                4: np.array([0., 0.])},
               {0: np.array([1, 1]),
                1: np.array([-1, 1]),
                2: np.array([2, 0]),
                3: np.array([-2, 0]),
                4: np.array([0, 0])},
               {0: np.array([0, 0]),
                1: np.array([1, 2]),
                2: np.array([2, 0]),
                3: np.array([1, 3]),
                4: np.array([1, 1])},
               {0: np.array([0.5, 1-np.sqrt(3)/2]),
                1: np.array([1.5, 1+np.sqrt(3)/2]),
                2: np.array([1.5, 1-np.sqrt(3)/2]),
                3: np.array([0.5, 1+np.sqrt(3)/2]),
                4: np.array([1, 1])},
               {0: np.array([0, 0]),
                1: np.array([1, 1+np.sqrt(2)/2]),
                2: np.array([2, 0]),
                3: np.array([1, 1-0.6]),
                4: np.array([1, 1])},
               {0: np.array([0, 0]),
                1: np.array([1, 1+np.sqrt(2)/2]),
                2: np.array([2, 0]),
                3: np.array([1, 1-0.6]),
                4: np.array([1, 1])},
               {0: np.array([0, 1]),
                1: np.array([np.sin(2*2*np.pi/5), np.cos(2*2*np.pi/5)]),
                2: np.array([np.sin(4*2*np.pi/5), np.cos(4*2*np.pi/5)]),
                3: np.array([np.sin(1*2*np.pi/5), np.cos(1*2*np.pi/5)]),
                4: np.array([np.sin(3*2*np.pi/5), np.cos(3*2*np.pi/5)])},
               {0: np.array([0, 1]),
                1: np.array([np.sin(2*2*np.pi/5), np.cos(2*2*np.pi/5)]),
                2: np.array([np.sin(4*2*np.pi/5), np.cos(4*2*np.pi/5)]),
                3: np.array([np.sin(1*2*np.pi/5), np.cos(1*2*np.pi/5)]),
                4: np.array([np.sin(3*2*np.pi/5), np.cos(3*2*np.pi/5)])},
               {0: np.array([0, 0]),
                1: np.array([0.5, 1+np.sqrt(3)/2]),
                2: np.array([1, 0]),
                3: np.array([0, 1]),
                4: np.array([1, 1])},
               {0: (3/np.sqrt(3))*np.array([0.5, np.sqrt(3)/5]),
                1: (3/np.sqrt(3))*np.array([0, 2*np.sqrt(3)/5]),
                2: (3/np.sqrt(3))*np.array([1., 0.]),
                3: (3/np.sqrt(3))*np.array([0.5, np.sqrt(3)/2]),
                4: (3/np.sqrt(3))*np.array([0., 0.])},
               {0: (3/np.sqrt(3))*np.array([1., 0.]),
                1: (3/np.sqrt(3))*np.array([0.5, np.sqrt(3)/2]),
                2: (3/np.sqrt(3))*np.array([0., 0.]),
                3: (3/np.sqrt(3))*np.array([1, 2*np.sqrt(3)/5]),
                4: (3/np.sqrt(3))*np.array([0.5, np.sqrt(3)/5])},
               {0: (3/np.sqrt(3))*np.array([1., 0.]),
                1: (3/np.sqrt(3))*np.array([0.5, np.sqrt(3)/2]),
                2: (3/np.sqrt(3))*np.array([0., 0.]),
                3: (3/np.sqrt(3))*np.array([1, 2*np.sqrt(3)/5]),
                4: (3/np.sqrt(3))*np.array([0.5, np.sqrt(3)/5])},
               {0: (3/np.sqrt(3))*np.array([1., 0.]),
                1: (3/np.sqrt(3))*np.array([0.2, 0.1+np.sqrt(3)/5]),
                2: (3/np.sqrt(3))*np.array([-0.25, 0.]),
                3: (3/np.sqrt(3))*np.array([0.25, np.sqrt(3)/2]),
                4: (3/np.sqrt(3))*np.array([0.5, np.sqrt(3)/5])},
               {0: np.array([0, 1]),
                1: np.array([np.sin(2*2*np.pi/5), np.cos(2*2*np.pi/5)]),
                2: np.array([np.sin(4*2*np.pi/5), np.cos(4*2*np.pi/5)]),
                3: np.array([np.sin(1*2*np.pi/5), np.cos(1*2*np.pi/5)]),
                4: np.array([np.sin(3*2*np.pi/5), np.cos(3*2*np.pi/5)])}]}

def black_or_white_text(color: tuple):
    """
    Determines whether it's more appropriate
    to write using black or white text on the
    given color.

    From: https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color

    Args:
        color (tuple): (R, G, B) between 0 and 1 each

    Returns:
        True for black text, False for white text
    """
    thresh = color[0]*0.299 + color[1]*0.587 + color[2]*0.114
    # 186/255
    return thresh > 0.729


def draw_circuit_graph(circuit: list, edges: list, gtype: str = "component",
                       out="",
                       node_size: float = 10000, scale: float = 6,
                       font_size: int = 30) -> Union[plt.Figure, None]:
    """
    Draws the port or component graph representation of a given circuit.

    This function visualizes a quantum circuit as either a **port graph** or a **component graph** 
    using NetworkX and Matplotlib. The generated graph can be displayed interactively 
    or saved as an image file.

    Parameters
    ----------
    circuit : list of list of str
        A nested list representing the circuit elements.  
        Example: `[['C'], ['C'], ['L'], ['C', 'J']]`
    edges : list of tuple of int
        A list of edge connections specifying how circuit elements are connected.  
        Example: `[(0,1), (1,2), (2,3), (3,0)]`
    gtype : str, optional
        Type of graph to draw. Options are:
        - `'component'` : Draws a component-level circuit graph.
        - `'port'` : Draws a port-level circuit graph.  
        Default is `'component'`.
    out : str, optional
        Filename (with extension) to save the plot.  
        If `""` (empty string), the graph is displayed interactively.  
        Default is `"circuit_graph.png"`.
    node_size : float, optional
        Size parameter for nodes in the plotted graph.
    scale : float, optional
        Scaling factor for overall spacing of nodes.
    font_size : int, optional
        Font size for node labels.

    Returns
    -------
    matplotlib.figure.Figure or None
        - If `out=""`, returns a Matplotlib figure object for interactive viewing.
        - Otherwise, saves the figure to `out` and returns `None`.

    Examples
    --------
    >>> draw_circuit_graph(
    >>>    circuit=[['C'], ['C'], ['L'], ['C', 'J']],
    >>>    edges=[(0,1), (1,2), (2,3), (3,0)],
    >>>    gtype="component",
    >>>    out="circuit.png"
    >>> )
    """

    # Get the layout and scale it
    fig = plt.figure()
    fig.set_size_inches(fig.get_size_inches()*scale)
    if gtype == "component":
        G = red.convert_circuit_to_component_graph(circuit, edges)
    else:
        G = red.convert_circuit_to_port_graph(circuit, edges)
    if nx.is_planar(G):
        pos = nx.planar_layout(G)
    else:
        pos = nx.spring_layout(G)
    for n in pos:
        pos[n] = pos[n]*scale

    # Use a colormap to generate a color for each node
    # And use that color to decide on white vs. black text for
    # each node
    node_color_map = nx.get_node_attributes(G, "color")
    node_color = np.array([node_color_map[n] for n in G.nodes()]).astype(float)
    node_color /= np.max(node_color)
    cmap = matplotlib.cm.get_cmap('viridis')
    black_text_nodes = {}
    white_text_nodes = {}

    node_color_mapped = []
    for i, n in enumerate(G.nodes()):
        color = cmap(node_color[i])
        node_color_mapped.append(color)
        if black_or_white_text(color):
            black_text_nodes[n] = pos[n]
        else:
            white_text_nodes[n] = pos[n]

    nx.draw(G, pos, node_color=node_color, node_size=node_size)
    # Draw black text nodes
    Gk = nx.Graph()
    for n in black_text_nodes:
        Gk.add_node(n)
    nx.draw_networkx_labels(Gk, black_text_nodes,
                            font_color='k', font_size=font_size)
    # Draw white text nodes
    Gw = nx.Graph()
    for n in white_text_nodes:
        Gw.add_node(n)
    nx.draw_networkx_labels(Gw, white_text_nodes,
                            font_color='w', font_size=font_size)

    if out != "":
        plt.savefig(out)
        plt.close()

    else:
        return fig


def print_bare_graphs(all_graphs: list):
    """Plots unlabeled graphs from graph list returned
    from get_graphs_from_file() function

    Args:
        all_graphs (list): a list of unlabeled graphs
    """
    for i, G in enumerate(all_graphs):
        draw_basegraph(G, f"graph index: {i}")


def draw_all_basegraphs(base_path: str, n_start: int = 2, n_end: int = 4):
    """
    Draws all the basegraphs for nodes with number
    n_start through n_end.

    Args:
        base_path (str): folder to save images in
        n_start (int, optional): start drawing graphs
                                 for this number of nodes
        n_end (int, optional): stop drawing graphs for this
                               number of nodes (inclusive)

    Returns:
        dict: positions of nodes for each plot
    """
    pos = {}
    for n_nodes in range(n_start, n_end + 1):
        pos[n_nodes] = []
        all_graphs = utils.get_basegraphs(n_nodes)
        for i, G in enumerate(all_graphs):
            fname = str(Path(base_path, f"n{n_nodes}_g{i}.svg"))
            title = f"{n_nodes} nodes, graph {i}"
            f, p = draw_basegraph(G, title, fname)
            for k in p:
                p[k] = np.round(p[k], 2)
            pos[n_nodes].append(p)

    return pos


def draw_basegraph(G: nx.Graph, title: str = "",
                   savename: str = None, **kwargs):
    """Plots unlabeled graphs from graph list returned
    from get_graphs_from_file() function

    Args:
        G (nx.graph): networkx graph to plot
        title (str): title for plot
        savename (str): location to save the plots
        pos (dict): dictionary of node labels -> positions

    Returns:
        tuple: figure, positioning (x, y) of nodes
    """
    edges = [x for x in G.edges()]
    n_nodes = G.number_of_nodes()
    graph_index = utils.edges_to_graph_index(edges)
    if n_nodes in G_POS_BG:
        def_pos = G_POS_BG[n_nodes][graph_index]
    else:
        if nx.is_planar(G):
            def_pos = nx.planar_layout(G)
        else:
            def_pos = nx.spring_layout(G)
    pos = kwargs.get("pos", def_pos)

    f = plt.figure(figsize=(3, 3))
    plt.tight_layout()
    nx.draw_networkx(G, pos=pos, node_color="#FFFFFF", linewidths=1, width=2, edgecolors="#000000",
                     with_labels=False, node_size=800)
    plt.gca().set_axis_off()
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, bbox_inches = "tight")
    return f, pos


def draw_all_qubits(file: str, n_nodes: int, out_dir: str,
                    layout: str = 'fixed', format: str = ".svg"):

    if n_nodes < 5:
        scale = 4.0
    else:
        scale = 4.0 + n_nodes - 3
    # So plots don't pop up
    matplotlib.use('agg')
    df = utils.get_unique_qubits(file, n_nodes)
    for uid, row in tqdm(df.iterrows(), total=df.shape[0]):
        circuit = row.circuit
        edges = row.edges
        graph_index = row.graph_index
        viz.draw_circuit_diagram(circuit, edges,
                                 out=Path(out_dir, f"{uid}{format}"),
                                 layout=layout, graph_index=graph_index,
                                 scale=scale)
        plt.close()


def draw_circuit_diagram(circuit: list, edges: list,
                         out: str = "",
                         scale: float = 4.0, layout: str = 'fixed',
                         spread: float = 2/5, graph_index: int = None) -> None:
    """
    Draw the circuit diagram using `schemdraw`.

    For parallel elements, connections are split 1/4 of the way along and fan out 
    to display parallel elements. Non-planar graphs are not adjusted to avoid overlap.

    Parameters
    ----------
    circuit : list
        A list of element labels for the desired circuit.  
        Example: ``[["J"], ["L", "J"], ["C"]]``.
    edges : list
        A list of edge connections for the desired circuit.  
        Example: ``[(0,1), (0,2), (1,2)]``.
    out : str
        Filename to save the plot. If an empty string (``""``) is provided, 
        the plot is displayed interactively.
    scale : float
        Scaling factor for the networkx positions to spread out the plots if needed.
    layout : str
        Options for graph layouts.  
        **"spring"** produces aesthetically pleasing circuits but may lead to 
        overlapping elements, even for planar graphs.
    spread : float
        Fraction of edge length used to fan out parallel components.
    graph_index : int
        Graph number for a fixed layout. Can also be inferred from `edges`.

    Returns
    -------
    None
        Displays the plot interactively if `out == ""`.

    Examples
    --------
    >>> draw_circuit_diagram(
    >>>     circuit=[["J"], ["L", "J"], ["C"]],
    >>>     edges=[(0,1), (0,2), (1,2)],
    >>>     out="circuit.png",
    >>>     layout="spring",
    >>>     scale=1.5
    >>> )
    """
    edges = utils.zero_start_edges(edges)

    elem_dict = {
        'C': {'default_unit': 'GHz', 'default_value': 0.2},
        'L': {'default_unit': 'GHz', 'default_value': 1.0},
        'J': {'default_unit': 'GHz', 'default_value': 5.0},
        }

    # Get rid of any labels on elements
    new_circuit = []
    for elems in circuit:
        new_elems = []
        for elem in elems:
            for base_elem in elem_dict:
                if base_elem in elem:
                    new_elems.append(base_elem)
                    continue
        new_circuit.append(tuple(new_elems))
    circuit = new_circuit

    params = utils.gen_param_dict(circuit, edges, elem_dict)
    G = utils.convert_circuit_to_graph(circuit, edges,
                                       params=params)

    # Get graph index
    if graph_index is None and layout == "fixed":
        graph_index = utils.edges_to_graph_index(edges)

    # Get layout of vertices
    if layout == 'planar':
        if nx.is_planar(G):
            pos = nx.planar_layout(G)
        else:
            print("Not a planar graph... reverting to spring layout")
            pos = nx.spring_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'fixed':
        pos = G_POS[G.number_of_nodes()][graph_index]

    # Scale
    scaled_pos = {}
    for k in pos:
        scaled_pos[k] = pos[k]*scale

    # Define the circuit elements
    elem_bank = {
        'C': lambda: elm.Capacitor(),
        'L': lambda: elm.Inductor2(),
        'J': lambda: elm.Josephson()
    }

    # Calculate minimum pairwise distance for plotting
    pdist_mat = pdist(np.vstack([x for x in scaled_pos.values()]))
    min_r = np.min(pdist_mat[pdist_mat > 0])

    with schemdraw.Drawing() as d:
        for n0 in G.nodes():
            # Draw a dot at every node
            d.add(DotCustom(radius=0.1, fill="#FFFFFF", color="#000000",
                            lw=1).at(scaled_pos[n0]))
            # d.add(schemdraw.segments.SegmentCircle(scaled_pos[n0],
            #                                        radius=0.2,
            #                                        color="#000000",
            #                                        fill="#FFFFFF",
            #                                        lw=0.01))
            for n1 in G.nodes():
                if n0 != n1 and n0 < n1 and G.has_edge(n0, n1):
                    edgesBetweenNodes = G[n0][n1]
                    # Single element
                    nEdges = len(edgesBetweenNodes)
                    if nEdges == 1:
                        edge = edgesBetweenNodes[0]
                        # Get the endpoints and specific circuit element
                        # that correspond to this edge
                        x0 = scaled_pos[n0]
                        x1 = scaled_pos[n1]
                        d.add(elem_bank[edge['element']]().endpoints(x0, x1))
                    # Parallel Elements
                    else:
                        # Draw a line perpendicular to the connection
                        # Split way there and spread out spread
                        # out and do them in parallel
                        split = 1/4
                        x0 = scaled_pos[n0]
                        x1 = scaled_pos[n1]

                        # Unit vectors along displacement and
                        # perpendicular to it
                        rhat = x1-x0
                        r = np.linalg.norm(rhat)
                        rhat /= r
                        norm = np.array([-rhat[1], rhat[0]])
                        perp_step = (spread*min_r/nEdges)*norm

                        # Anchor points for splitting
                        adj = split*rhat*(r-min_r)
                        a0 = x0+split*rhat*r+adj
                        a1 = x1-split*rhat*r-adj
                        d.add(elm.Wire().at(x0).to(a0))
                        d.add(elm.Wire().at(x1).to(a1))

                        # Start at 0 and go out integer
                        # steps if odd numberof edges.
                        # For even number of edges do 1/2 integer steps
                        if nEdges % 2 == 0:
                            maxSpread = (perp_step*(nEdges/2-1/2))
                            d.add(elm.Wire().at(a0).to(a0-maxSpread))
                            d.add(elm.Wire().at(a0).to(a0+maxSpread))
                            d.add(elm.Wire().at(a1).to(a1-maxSpread))
                            d.add(elm.Wire().at(a1).to(a1+maxSpread))
                            for edgeNum in range(int(nEdges/2)):
                                edge1 = edgesBetweenNodes[edgeNum]
                                edge2 = edgesBetweenNodes[nEdges-1-edgeNum]
                                d.add(elem_bank[edge1['element']]().endpoints(
                                    a0+perp_step*(edgeNum+1/2),
                                    a1+perp_step*(edgeNum+1/2)))
                                d.add(elem_bank[edge2['element']]().endpoints(
                                    a0-perp_step*(edgeNum+1/2),
                                    a1-perp_step*(edgeNum+1/2)))
                        else:
                            maxSpread = (perp_step*(nEdges-1)/2)
                            d.add(elm.Wire().at(a0).to(a0-maxSpread))
                            d.add(elm.Wire().at(a0).to(a0+maxSpread))
                            d.add(elm.Wire().at(a1).to(a1-maxSpread))
                            d.add(elm.Wire().at(a1).to(a1+maxSpread))
                            # 0th one straight across
                            d.add(elem_bank[edgesBetweenNodes[0]
                                  ['element']]().endpoints(a0, a1))

                            # Others mirrored across -- work in pairs
                            for edgeNum in range(1, int((nEdges-1)/2 + 1)):
                                edge1 = edgesBetweenNodes[edgeNum]
                                edge2 = edgesBetweenNodes[nEdges-edgeNum]
                                d.add(elem_bank[edge1['element']]().endpoints(
                                    a0+perp_step*edgeNum,
                                    a1+perp_step*edgeNum))
                                d.add(elem_bank[edge2['element']]().endpoints(
                                    a1-perp_step*edgeNum,
                                    a0-perp_step*edgeNum))
        if out != "":
            d.save(out)
    if out != "":
        plt.close()
    else:
        plt.show()


class DotCustom(schemdraw.elements.Element):
    ''' Connection Dot

        Keyword Args:
            radius: Radius of dot [default: 0.075]
            open: Draw as an open circle [default: False]
    '''
    _element_defaults = {
        'radius': 0.075,
        'open': False}
    def __init__(self, *,
                 radius,
                 fill,
                 **kwargs):
        super().__init__(**kwargs)
        fill = fill
        self.anchors['start'] = (0, 0)
        self.anchors['center'] = (0, 0)
        self.anchors['end'] = (0, 0)
        self.elmparams['drop'] = (0, 0)
        self.elmparams['theta'] = 0
        self.elmparams['zorder'] = 4
        self.elmparams['fill'] = fill
        self.segments.append(schemdraw.segments.SegmentCircle((0, 0),
                                                              self.params['radius'],
                                                              **kwargs))
