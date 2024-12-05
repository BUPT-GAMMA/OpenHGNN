import tkinter as tk
from tkinter import ttk

from openhgnn.experiment import Experiment
import threading
import sys

# 定义一个类来捕获stdout并输出到tkinter的Text控件
class OutputRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.update_idletasks()  # 强制刷新GUI界面

    def flush(self):
        pass  # 如果需要刷新缓冲区，可以在这里实现


def run_experiment_in_thread():
    # 在新线程中运行 Experiment，这样在模型训练过程中，GUI窗口也可以同时更新输出内容
    threading.Thread(target=run_experiment, daemon=True).start()


def run_experiment():
    model = model_var.get()
    task = task_var.get()
    dataset = dataset_var.get()
    gpu = int(gpu_var.get())  # 将字符串转换为整数

    use_distributed = use_distributed_var.get()
    use_best_config = use_best_config_var.get()
    load_from_pretrained = load_from_pretrained_var.get()
    use_database = use_database_var.get()
    mini_batch_flag = mini_batch_flag_var.get()
    graphbolt = graphbolt_var.get()

    # 清空输出框内容，
    output_widget.delete(1.0, tk.END)
    output_widget.insert(tk.END, "OpenHGNN  main.py 开始运行  \n")
    output_widget.update_idletasks()


    # 重定向 stdout
    sys.stdout = OutputRedirector(output_widget)

    #   把 GUI界面中的10个参数，传入 experiment对象
    experiment = Experiment(model=model, dataset=dataset, task=task, gpu=gpu,
                            use_best_config=use_best_config, load_from_pretrained=load_from_pretrained,
                            mini_batch_flag=mini_batch_flag, use_distributed=use_distributed,
                            graphbolt=graphbolt,
                            output_widget=output_widget)
    
    experiment.run()

    print("pipeline finished")

# 创建主窗口
root_main_window = tk.Tk()
root_main_window.title("Experiment Configuration")

# 创建控件变量
model_var = tk.StringVar(value='RGCN')
task_var = tk.StringVar(value='node_classification')
dataset_var = tk.StringVar(value='acm4GTN')
gpu_var = tk.StringVar(value='0')  # GPU 参数仍是字符串输入，但会在运行时转换为整数

use_distributed_var = tk.BooleanVar()
use_best_config_var = tk.BooleanVar()
load_from_pretrained_var = tk.BooleanVar()
use_database_var = tk.BooleanVar()
mini_batch_flag_var = tk.BooleanVar()
graphbolt_var = tk.BooleanVar()


# 4个用于输入文本的控件
tk.Label(root_main_window, text="Model").grid(row=0, column=0, padx=10, pady=5)
model_entry = ttk.Entry(root_main_window, textvariable=model_var)
model_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root_main_window, text="Task").grid(row=1, column=0, padx=10, pady=5)
task_entry = ttk.Entry(root_main_window, textvariable=task_var)
task_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root_main_window, text="Dataset").grid(row=2, column=0, padx=10, pady=5)
dataset_entry = ttk.Entry(root_main_window, textvariable=dataset_var)
dataset_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root_main_window, text="GPU").grid(row=3, column=0, padx=10, pady=5)
gpu_entry = ttk.Entry(root_main_window, textvariable=gpu_var)
gpu_entry.grid(row=3, column=1, padx=10, pady=5)

#   6个用于勾选的 bool型控件
tk.Checkbutton(root_main_window, text="Use Distributed Training", variable=use_distributed_var).grid(row=4, column=0, columnspan=2, pady=5)
tk.Checkbutton(root_main_window, text="Use Best Config", variable=use_best_config_var).grid(row=5, column=0, columnspan=2, pady=5)
tk.Checkbutton(root_main_window, text="Load from Pretrained", variable=load_from_pretrained_var).grid(row=6, column=0, columnspan=2, pady=5)
tk.Checkbutton(root_main_window, text="Use Database", variable=use_database_var).grid(row=7, column=0, columnspan=2, pady=5)
tk.Checkbutton(root_main_window, text="Mini Batch Mode", variable=mini_batch_flag_var).grid(row=8, column=0, columnspan=2, pady=5)
tk.Checkbutton(root_main_window, text="Use Graphbolt", variable=graphbolt_var).grid(row=9, column=0, columnspan=2, pady=5)

# 创建输出窗口
output_widget = tk.Text(root_main_window, height=15, width=70, wrap='word')
output_widget.grid(row=11, column=0, columnspan=2, padx=10, pady=10)

# 创建运行按钮，直接调用 run_experiment
run_button = ttk.Button(root_main_window, text="Run Experiment", command=run_experiment_in_thread)
run_button.grid(row=10, column=0, columnspan=2, pady=10)


# 运行主循环
root_main_window.mainloop()
