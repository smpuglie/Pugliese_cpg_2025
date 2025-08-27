import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
# from src import vncNet

def get_module_colors():
    swingColorMap = sns.blend_palette(["#B33B24","#FF8833","#F2C649"],4)
    stanceColorMap = sns.blend_palette(["#492c6d","#7070CC","#93CCEA"],4)
    colors = {
        "coxa swing": swingColorMap[0],
        "coxa stance": stanceColorMap[0],
        "femur/tr extend": swingColorMap[1],
        "femur/tr flex": stanceColorMap[1],
        "femur reductor": "#DB5079",
        "tibia extend": swingColorMap[2],
        "tibia flex": stanceColorMap[2],
        "tibia flex A": stanceColorMap[2],
        "tibia flex B": stanceColorMap[2],
        "tibia flex C": stanceColorMap[2],
        "tarsus control": swingColorMap[3],
        "substrate grip": stanceColorMap[3],
        np.nan: "#808080",
    }
    return colors

def sort_motor_modules(neuronData,colName="motor module"):
    """sorts motor modules from proximal to distal leg segments"""
    neuronData[colName] = pd.Categorical(neuronData[colName],categories=["coxa swing", "coxa stance", "femur/tr extend", "femur/tr flex",
                                                                    "femur reductor", "tibia extend", "tibia flex",  "tibia flex A", "tibia flex B",
                                                                    "tibia flex C", "substrate grip", "tarsus control"], ordered=True)
    return neuronData.sort_values(by=colName)

def get_active_data(R,neuronData):
    return neuronData.loc[neuronData.index[np.where(np.max(R,1)>0)]]

def neuron_plot_labels(neuronData,fanc=False):
    if fanc:
        # label = [f'{neuronData.loc[i,"w_type"].replace("_"," ")} ({neuronData.loc[i,"pt_root_id"]})' for i in neuronData.index]
        label = [f'{neuronData.loc[i,"w_type"].replace("_"," ")}' for i in neuronData.index]
    else:
        label = [f'{neuronData.loc[i,"somaSide"][0]} {neuronData.loc[i,"type"]} ({neuronData.loc[i,"bodyId"]})' for i in neuronData.index]
        # label = [f'{neuronData.loc[i,"type"]}' for i in neuronData.index]

    return label

def plot_R_heatmap(R,neuronData,figsize=None,cmap="viridis",activeOnly=False,fanc=False,ax=None):
    """plots a heatmap of rate data (R) over time for the neurons in neuronData"""
    if "motor module" in neuronData:
        # make it so that motor module sorts from proximal to distal leg
        neuronData = sort_motor_modules(neuronData)
        
    sortOrder = [col for col in ["somaSide","motor module","type"] if col in neuronData] #TODO maybe allow for some flexibility here...
    neuronData.sort_values(by=sortOrder,inplace=True)
    sortedIdxs = neuronData.index
    sortedR = R[sortedIdxs]
    if activeOnly:
        neuronData = get_active_data(sortedR,neuronData)
        #  neuronData.loc[neuronData.index[np.where(np.sum(sortedR,1)>0)]]
        sortedIdxs = neuronData.index
        sortedR = R[sortedIdxs]
    if figsize is None:
        figsize = (6,len(neuronData)/8)
    sns.heatmap(R[sortedIdxs],cmap=cmap,cbar_kws={"label":"firing rate"})
    ax = plt.gca()

    ylabels = neuron_plot_labels(neuronData,fanc=fanc)
    ax.set_yticks(np.arange(0,len(sortedIdxs)),labels=ylabels,fontsize=8,rotation=0)
    [ax.get_yticklabels()[i].set_color("navy") for i in np.where(neuronData["step contribution"]=="stance")[0]]
    [ax.get_yticklabels()[i].set_color("orange") for i in np.where(neuronData["step contribution"]=="swing")[0]]
    ax.set_xticks([])

    return ax

def plot_R_traces(R,neuronData,cmap="tab10",figsize=(6,4),activeOnly=False,colors=None,fanc=False,ax=None):

    idxs = neuronData.index
    # R = R[idxs]
    # activeData = get_active_data(R[mnTable.index],mnTable)
    if activeOnly:
        neuronData = get_active_data(R[idxs],neuronData)
        #  neuronData.loc[neuronData.index[np.where(np.sum(sortedR,1)>0)]]
        idxs = neuronData.index
    R = R[idxs]


    if ax is None:
        ax = plt.gca()

    if colors is None:
        try:
            colors = plt.colormaps[cmap].resampled(len(neuronData)+1)
        except:
            colors = cmap.resampled(len(neuronData)+1)
        for i in range(len(idxs)):
            ax.plot(R[i],c=colors(i),label=neuron_plot_labels(neuronData.iloc[[i]],fanc=fanc)[0])
    else:
        for i in range(len(idxs)):
            ax.plot(R[i],c=colors[i],label=neuron_plot_labels(neuronData.iloc[[i]],fanc=fanc)[0])
    
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_ylabel("firing rate")
    ax.set_facecolor("None")
    ax.legend(loc="upper left",bbox_to_anchor=[1,1],edgecolor="None",facecolor="None")
    plt.gcf().set_figheight(figsize[1])
    plt.gcf().set_figwidth(figsize[0])

    return ax

def plot_R_traces_stacked_by_module(R,neuronData,colorMapper=None,figsize=(6,4),activeOnly=False,fanc=False,ax=None,space=None,start=0,alpha=1):

    idxs = neuronData.index
    # R = R[idxs]
    # activeData = get_active_data(R[mnTable.index],mnTable)
    if activeOnly:
        neuronData = get_active_data(R[idxs],neuronData)
        #  neuronData.loc[neuronData.index[np.where(np.sum(sortedR,1)>0)]]
        idxs = neuronData.index
    R = R[idxs]

    nTraces = len(idxs)
    if space is None:
        space = np.ceil(np.max(R)/5)*5+5

    if ax is None:
        ax = plt.gca()

    if colorMapper is None:
        # default
        colorMapper = get_module_colors()
            
    for i in range(nTraces):
        try:
            ax.plot(R[i]+space*(nTraces-i-1)+start,c=colorMapper[neuronData["motor module"].iloc[i]],
                    label=neuron_plot_labels(neuronData.iloc[[i]],fanc=fanc)[0],alpha=alpha,linewidth=0.75)
        except:
            ax.plot(R[i]+space*(nTraces-i-1)+start,c="#808080",label=neuron_plot_labels(neuronData.iloc[[i]],fanc=fanc)[0],alpha=alpha,linewidth=0.75)
    
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_ylabel("firing rate (Hz)")
    ax.set_facecolor("None")
    ax.legend(loc="upper left",bbox_to_anchor=[1,1],edgecolor="None",facecolor="None")
    xlimits = ax.get_xlim()
    plt.vlines(xlimits[0],0,space,"k")
    plt.yticks([0,space])
    plt.xlim(xlimits)
    ax.spines[["left","top","right"]].set_visible(False)
    # ax.grid(axis="x",color="lightgrey",linestyle="--")
    fig = plt.gcf()
    fig.set_figheight(0.5*nTraces)

    # plt.gcf().set_figheight(figsize[1])
    plt.gcf().set_figwidth(figsize[0])

    return ax

def plot_R_traces_stacked_2(R,neuronData,cmap="tab10",figsize=(6,4),activeOnly=False,colors=None,fanc=False,ax=None,space=None,start=0,alpha=1):

    idxs = neuronData.index
    # R = R[idxs]
    # activeData = get_active_data(R[mnTable.index],mnTable)
    if activeOnly:
        neuronData = get_active_data(R[idxs],neuronData)
        #  neuronData.loc[neuronData.index[np.where(np.sum(sortedR,1)>0)]]
        idxs = neuronData.index
    R = R[idxs]

    nTraces = len(idxs)
    if space is None:
        space = np.ceil(np.max(R)/5)*5+5

    if ax is None:
        ax = plt.gca()

    if colors is None:
        try:
            colors = plt.colormaps[cmap].resampled(len(neuronData)+1)
        except:
            colors = cmap.resampled(len(neuronData)+1)
        for i in range(nTraces):
            ax.plot(R[i]+space*(nTraces-i-1)+start,c=colors(i),label=neuron_plot_labels(neuronData.iloc[[i]],fanc=fanc)[0],alpha=alpha,linewidth=0.75)
    else:
        for i in range(nTraces):
            ax.plot(R[i]+space*(nTraces-i-1)+start,c=colors[i],label=neuron_plot_labels(neuronData.iloc[[i]],fanc=fanc)[0],alpha=alpha,linewidth=0.75)
    
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_ylabel("firing rate")
    ax.set_facecolor("None")
    ax.legend(loc="upper left",bbox_to_anchor=[1,1],edgecolor="None",facecolor="None")
    plt.gcf().set_figheight(figsize[1])
    plt.gcf().set_figwidth(figsize[0])

    return ax

def plot_R_traces_stacked(R,neuronData,cmap="tab10",figsize=(5,5),activeOnly=False,colorList=None,hspace=-0.4,ymax=None,fanc=False,fig=None):

    idxs = neuronData.index
    # R = R[idxs]
    # activeData = get_active_data(R[mnTable.index],mnTable)
    if activeOnly:
        neuronData = get_active_data(R[idxs],neuronData)
        #  neuronData.loc[neuronData.index[np.where(np.sum(sortedR,1)>0)]]
        idxs = neuronData.index


    if colorList is not None:
        cmap = colors.LinearSegmentedColormap.from_list("minimalINs",colorList,len(colorList))

    plotDf = pd.DataFrame(R[idxs],index = neuron_plot_labels(neuronData,fanc=fanc))
    if fig is None:
        fig = plt.figure(figsize=figsize)
    try:
        axes = plotDf.transpose().plot(subplots=True,colormap=cmap,figsize=figsize,linewidth=2)
    except: # no data to plot
        axes = [fig.gca()]        
        return fig, axes
    
    if ymax is None:
        ylims = [ax.get_ylim()[1] for ax in axes]
        ymax = np.ceil(np.max(ylims)/5)*5+5
    for ax in axes:
        ax.grid(True,axis="x",alpha=0.2)
        ax.legend(loc="upper left",bbox_to_anchor=[1,0.5],edgecolor="None",facecolor="None")
        ax.set_xticks([])
        ax.set_xticks([],minor=True)
        ax.set_yticks([])
        # ax.set_yticks([0],labels="")
        # ax.tick_params("y",colors=ax.get_lines()[0].get_color())
        ax.set_ylim([-ymax/20,ymax])
        ax.spines[:].set_visible(False)
        ax.set_facecolor("None")

    for o in fig.findobj():
        o.set_clip_on(False)

    plt.subplots_adjust(hspace=hspace)
    axes[-1].set_yticks([0,int(ymax)],labels=[0,ymax])
    axes[-1].set_ylabel("firing rate (Hz)")
    # plt.tight_layout()

    return fig, axes

def add_tAxis(ax,T,nTicks,decimalPlaces=2,rotation=0,defaultLen=0):
    if len(ax.collections) > defaultLen:
        try:
            nTimeSteps = np.size(ax.collections[0].get_array().data,1)
        except:
            nTimeSteps = len(ax.lines[0].get_xdata()) # IDK

    elif len(ax.lines) > defaultLen:
        nTimeSteps = len(ax.lines[0].get_xdata())
    else:
        ax.set_xlabel("no data",fontweight="bold",color="r")
        ax.set_facecolor("#eee")
        return ax
        
    dt = T / nTimeSteps
    # dataTAxis = vncNet.create_time_axis(T,dt)
    dataTAxis = np.array(np.arange(0, T, dt))
    

    tickIdxs = np.arange(0,nTimeSteps,int(nTimeSteps/nTicks)).astype(int)
    ax.set_xticks(tickIdxs,labels=np.round(dataTAxis[tickIdxs],decimalPlaces),rotation=rotation)
    ax.set_xlabel("Time (s)")
    ax.spines[["top","right"]].set_visible(False)

    return ax