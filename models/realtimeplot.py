import matplotlib.pyplot as plt
import numpy as np

fs = 256
E = 2
N = fs*E

def realtimeplot(I, signal):
    me = np.mean(signal)
    se = np.std(signal)
    signal = (signal - me) / se
    plt.ion()
    fig, ax = plt.subplots()
    x_data = range(N)
    y_data = [None] * N
    line, = ax.plot(x_data, y_data, label='Sinal em tempo real', color='green')
    line2, = ax.plot(x_data, y_data, color='green')
    line.set_xdata(x_data)
    line2.set_xdata(x_data)
    ax.set_xlim(0, N)
    ax.set_ylim(min(signal), max(signal))
    ax.set_title("Canal 1")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Sinal")
    plt.legend()

    for i in range(len(signal)):
        new_time = i % N

        if new_time == 0:
            e = i/N
            if(e in I):
                line.set_color(color='red')
            else:
                line.set_color(color='green')
            if(e-1 in I):
                line2.set_color(color='red')
            else:
                line2.set_color(color='green')
            z_data = y_data
            y_data = [None] * N

        y_data[new_time] = signal[i]
        z_data[new_time] = None

        line.set_ydata(y_data)
        line2.set_ydata(z_data)

        ax.relim()
        ax.autoscale_view()
        plt.pause(10/N)

    plt.ioff()
    plt.show()
