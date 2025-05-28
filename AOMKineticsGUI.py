import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from scipy.integrate import quad
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
import os
import math
import traceback

# 物理常数
R = 8.314  # 气体常数，J/(mol·K)
F = 96485.33289  # 法拉第常数，C/mol
h = 4.13568e-15  # 普朗克常数，eV·s
kB = 8.61689e-5  # 玻尔兹曼常数，eV/K
eV_to_J = 1.60218e-19  # 电子伏特到焦耳的转换因子
epsilon = 1e-6

class AOMKineticsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AOM Kinetic Simulation")
        self.root.geometry("1200x800")
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 11, 'bold'))

        # 初始化变量
        self.model_var = tk.StringVar(value="ER-AOM")
        self.kinetics_var = tk.StringVar(value="Butler-Volmer kinetics")
        self.bv_method_var = tk.StringVar(value="BEP")  # 新增变量
        self.chem_method_var = tk.StringVar(value="BEP")# 化学反应方法
        self.chem_method_var.trace_add("write", self.update_parameters)  # 切换时更新参数界面
        self.variable_var = tk.StringVar(value="η")
        self.delta_gw_var = tk.DoubleVar(value=0.8277)  # 新增△G_w变量
        self.start_var = tk.DoubleVar(value=-1.0)
        self.end_var = tk.DoubleVar(value=1.0)
        self.step_var = tk.DoubleVar(value=0.01)
        self.fixed_eta_var = tk.DoubleVar(value=0.5)
        self.fixed_ph_var = tk.DoubleVar(value=0)
        
        # 2D扫描参数
        self.eta_start_var = tk.DoubleVar(value=-1.0)
        self.eta_end_var = tk.DoubleVar(value=1.0)
        self.eta_step_var = tk.DoubleVar(value=0.01)
        self.ph_start_var = tk.DoubleVar(value=0)
        self.ph_end_var = tk.DoubleVar(value=14)
        self.ph_step_var = tk.DoubleVar(value=1)
        
        # 存储参数的Entry部件
        self.er_aom_bv_entries = []
        self.er_aom_marcus_entries = []
        self.er_aom_mg_entries = []  # 初始化MG参数容器
        self.lh_aom_bv_entries = []
        self.lh_aom_marcus_entries = []
        self.lh_aom_mg_entries = []  # 初始化MG参数容器
        
        # 创建界面
        self.create_main_layout()
        self.update_parameters()
        self.update_variable_controls()
        self.setup_plots()
        self.plot_window = None  # 用于跟踪当前绘图窗口
        self.current_fig = None  # 避免Matplotlib资源未释放
                
    def create_main_layout(self):
        # 主容器使用Grid布局
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ===== 左侧控制面板 =====
        left_panel = ttk.Frame(main_container, width=480)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # 参数框架
        param_frame = ttk.LabelFrame(left_panel, text="Parameters", padding="10")
        param_frame.pack(fill=tk.BOTH, expand=True)

        # 配置区域
        config_frame = ttk.Frame(param_frame)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # 模型选择
        model_frame = ttk.LabelFrame(config_frame, text="1. Model Selection", padding="10")
        model_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Radiobutton(model_frame, text="ER-AOM", variable=self.model_var,
                       value="ER-AOM", command=self.update_parameters).pack(anchor=tk.W)
        ttk.Radiobutton(model_frame, text="LH-AOM", variable=self.model_var,
                       value="LH-AOM", command=self.update_parameters).pack(anchor=tk.W)

        # 动力学公式选择
        kinetics_frame = ttk.LabelFrame(config_frame, text="2.1 Kinetics Formula(ECR)", padding="10")
        kinetics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        kinetics_options = ["Butler-Volmer kinetics", "Marcus kinetics", "Marcus-Gerischer kinetics"]
        for option in kinetics_options:
            ttk.Radiobutton(kinetics_frame, text=option, variable=self.kinetics_var,
                           value=option, command=self.update_parameters).pack(anchor=tk.W)
        # 新增：B-V方法选择子框架（仅当选择B-V时显示）
        self.bv_sub_frame = ttk.Frame(kinetics_frame)   
        ttk.Radiobutton(self.bv_sub_frame, text="BEP", variable=self.bv_method_var,
                        value="BEP", command=self.update_parameters).pack(side=tk.LEFT)
        ttk.Radiobutton(self.bv_sub_frame, text="Softplus", variable=self.bv_method_var,
                        value="Softplus", command=self.update_parameters).pack(side=tk.LEFT)
        
        # 在kinetics_frame下方添加新的化学反应方法选择框架
        self.chem_frame = ttk.LabelFrame(config_frame, text="2.2 Chemical Reaction Method(CR)", padding="10")
        self.chem_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.chem_method_var = tk.StringVar(value="BEP")  # 新增变量
        ttk.Radiobutton(self.chem_frame, text="BEP", variable=self.chem_method_var, value="BEP").pack(anchor=tk.W)
        ttk.Radiobutton(self.chem_frame, text="Softplus", variable=self.chem_method_var, value="Softplus").pack(anchor=tk.W)

        # 变量控制
        var_frame = ttk.LabelFrame(config_frame, text="3. Variable Control", padding="10")
        var_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Label(var_frame, text="Variable:").pack(anchor=tk.W)
        ttk.Radiobutton(var_frame, text="η (overpotential)", variable=self.variable_var,
                       value="η", command=self.update_variable_controls).pack(anchor=tk.W)
        ttk.Radiobutton(var_frame, text="pH", variable=self.variable_var,
                       value="pH", command=self.update_variable_controls).pack(anchor=tk.W)
        ttk.Radiobutton(var_frame, text="2D Scan (η & pH)", variable=self.variable_var,
                       value="2D", command=self.update_variable_controls).pack(anchor=tk.W)
        
        # 变量范围
        ttk.Label(var_frame, text="Range:").pack(anchor=tk.W, pady=(10, 0))
        range_frame = ttk.Frame(var_frame)
        range_frame.pack(fill=tk.X)
        ttk.Label(range_frame, text="Start:").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.start_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(range_frame, text="End:").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.end_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(range_frame, text="Step:").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.step_var, width=8).pack(side=tk.LEFT, padx=5)

        # 固定值框架
        self.fixed_frame = ttk.Frame(var_frame)
        self.fixed_frame.pack(fill=tk.X, pady=5)
        
        # 固定η控制
        self.fixed_eta_frame = ttk.Frame(self.fixed_frame)
        ttk.Label(self.fixed_eta_frame, text="Fixed η:").pack(side=tk.LEFT)
        ttk.Entry(self.fixed_eta_frame, textvariable=self.fixed_eta_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # 固定pH控制
        self.fixed_ph_frame = ttk.Frame(self.fixed_frame)
        ttk.Label(self.fixed_ph_frame, text="Fixed pH:").pack(side=tk.LEFT)
        ttk.Entry(self.fixed_ph_frame, textvariable=self.fixed_ph_var, width=8).pack(side=tk.LEFT, padx=5)

        # 2D扫描范围框架
        self.eta_2d_frame = ttk.Frame(var_frame)
        ttk.Label(self.eta_2d_frame, text="η Range:").pack(side=tk.LEFT)
        ttk.Entry(self.eta_2d_frame, textvariable=self.eta_start_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.eta_2d_frame, text="to").pack(side=tk.LEFT)
        ttk.Entry(self.eta_2d_frame, textvariable=self.eta_end_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.eta_2d_frame, text="step").pack(side=tk.LEFT)
        ttk.Entry(self.eta_2d_frame, textvariable=self.eta_step_var, width=8).pack(side=tk.LEFT, padx=5)

        self.ph_2d_frame = ttk.Frame(var_frame)
        ttk.Label(self.ph_2d_frame, text="pH Range:").pack(side=tk.LEFT)
        ttk.Entry(self.ph_2d_frame, textvariable=self.ph_start_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.ph_2d_frame, text="to").pack(side=tk.LEFT)
        ttk.Entry(self.ph_2d_frame, textvariable=self.ph_end_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.ph_2d_frame, text="step").pack(side=tk.LEFT)
        ttk.Entry(self.ph_2d_frame, textvariable=self.ph_step_var, width=8).pack(side=tk.LEFT, padx=5)

        # 参数容器
        self.param_frame_container = ttk.Frame(param_frame)
        self.param_frame_container.pack(fill=tk.BOTH, expand=True)

        # 按钮区域
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Calculate", command=self.calculate).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=10)

        # 结果表格
        results_frame = ttk.LabelFrame(left_panel, text="结果表格", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 创建带双滚动条的容器
        tree_container = ttk.Frame(results_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)

        # 创建Treeview组件
        self.tree = ttk.Treeview(
            tree_container,
            columns=[],
            show='headings',
            height=8,
            selectmode='extended'
        )

        # 创建垂直滚动条
        vsb = ttk.Scrollbar(
            tree_container,
            orient="vertical",
            command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=vsb.set)

        # 创建水平滚动条
        hsb = ttk.Scrollbar(
            tree_container,
            orient="horizontal",
            command=self.tree.xview
        )
        self.tree.configure(xscrollcommand=hsb.set)

        # 使用grid布局管理器
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # 配置容器网格权重
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        # 添加自动调整列宽功能
        def auto_resize_columns(event=None):
            for col in self.tree["columns"]:
                max_width = max(
                    tk.font.Font().measure(str(self.tree.set(item, col))) 
                    for item in self.tree.get_children()
                )
                self.tree.column(col, width=max_width + 20)

        self.tree.bind("<Configure>", auto_resize_columns)

        # ===== 右侧图表容器 =====
        self.right_container = ttk.Frame(main_container)
        self.right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)        

        # 配置权重分配
        main_container.columnconfigure(0, weight=0)  # 左侧固定宽度
        main_container.columnconfigure(1, weight=1)  # 右侧自适应
                       
    def setup_plots(self):
        """初始化右侧的三个图表"""
        # 创建滚动容器
        canvas = tk.Canvas(self.right_container, borderwidth=0)
        scrollbar = ttk.Scrollbar(self.right_container, 
                                orient="vertical", 
                                command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        # 配置滚动区域
        scroll_frame.bind("<Configure>", 
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 布局滚动组件
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 创建图表
        self.fig = plt.figure(figsize=(8, 10), dpi=80, constrained_layout=True)  # 调整尺寸
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1,1,1.2], hspace=0.8)
    
        self.ax_k = self.fig.add_subplot(gs[0])
        self.ax_lgr = self.fig.add_subplot(gs[1])
        self.ax_theta = self.fig.add_subplot(gs[2])

        # 嵌入图表
        self.canvas = FigureCanvasTkAgg(self.fig, master=scroll_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加水平滚动条
        h_scroll = ttk.Scrollbar(self.right_container,
                               orient="horizontal",
                               command=canvas.xview)
        canvas.configure(xscrollcommand=h_scroll.set)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

    def create_plot_window(self):
        """创建独立的绘图窗口"""
        # 关闭旧窗口（如果存在）
        if self.plot_window is not None:
            try:
                self.plot_window.destroy()
                plt.close(self.current_fig)  # 显式关闭Figure
            except:
                pass

        # 创建新窗口
        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title("仿真结果图表")
        self.plot_window.geometry("1200x800")

        # 创建Figure和Canvas（绑定到新窗口）
        self.current_fig = plt.figure(figsize=(10, 15), dpi=100)
        gs = self.current_fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5], hspace=0.7)
        self.ax_k = self.current_fig.add_subplot(gs[0])
        self.ax_lgr = self.current_fig.add_subplot(gs[1])
        self.ax_theta = self.current_fig.add_subplot(gs[2])

        # 嵌入Canvas到新窗口
        self.canvas = FigureCanvasTkAgg(self.current_fig, master=self.plot_window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

         # 添加工具栏（需先导入 NavigationToolbar2Tk）
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_window)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot_in_new_window(self):
        """在新窗口中更新图表"""
        try:
            if not hasattr(self, 'results_df') or self.results_df.empty:
                raise ValueError("无有效数据，请先进行计算")

            # 绘制图表
            self.draw_plots(self.ax_k, self.ax_lgr, self.ax_theta)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("绘图错误", f"图表生成失败：\n{str(e)}")
            traceback.print_exc()

    def draw_plots(self, ax_k, ax_lgr, ax_theta):
        """通用绘图逻辑（无Figure依赖）"""
        # 清除旧数据
        ax_k.clear()
        ax_lgr.clear()
        ax_theta.clear()

        # 获取变量数据
        variable = self.results_df.columns[0]
        x = self.results_df[variable]

        # 设置颜色主题
        colors = plt.cm.tab10.colors

        # 绘制速率常数
        k_cols = [c for c in self.results_df.columns if c.startswith('k') ]
        for idx, col in enumerate(k_cols):
            ax_k.plot(x, self.results_df[col], color=colors[idx%10], linewidth=1.5, label=col)
        ax_k.set_title("Rate Constants", fontsize=12, pad=10)
        ax_k.legend(fontsize=8, loc='upper right', framealpha=0.8)
        ax_k.grid(True, linestyle='--', alpha=0.6)

        # 绘制对数速率
        lgr_cols = [c for c in self.results_df.columns if c.startswith(('lg(r1)', 'lg(r21)', 'lg(r22)', 'lg(r5)'))]
        for idx, col in enumerate(lgr_cols):
            ax_lgr.plot(x, self.results_df[col], color=colors[idx%10], linewidth=1.5, label=col.replace('lg', 'log'))
        ax_lgr.set_title("Reaction Rates (log scale)", fontsize=12, pad=10)
        ax_lgr.legend(fontsize=8, loc='upper right', framealpha=0.8)
        ax_lgr.grid(True, linestyle='--', alpha=0.6)

        # 绘制表面覆盖度
        theta_cols = [c for c in self.results_df.columns if c.startswith('theta')]
        for idx, col in enumerate(theta_cols):
            ax_theta.plot(x, self.results_df[col], color=colors[idx%10], linewidth=1.5, label=col)
        ax_theta.set_title("Surface Coverage", fontsize=12, pad=10)
        ax_theta.legend(fontsize=8, loc='upper right', framealpha=0.8)
        ax_theta.grid(True, linestyle='--', alpha=0.6)

        # 设置公共坐标标签
        ax_theta.set_xlabel(variable, fontsize=10)

    def update_variable_controls(self):
        current_var = self.variable_var.get()
        self.fixed_eta_frame.pack_forget()
        self.fixed_ph_frame.pack_forget()
        self.eta_2d_frame.pack_forget()
        self.ph_2d_frame.pack_forget()
        
        if current_var == "η":
            self.fixed_ph_frame.pack(anchor=tk.W)
        elif current_var == "pH":
            self.fixed_eta_frame.pack(anchor=tk.W)
        elif current_var == "2D":
            self.eta_2d_frame.pack(fill=tk.X, pady=5)
            self.ph_2d_frame.pack(fill=tk.X, pady=5)

    def update_parameters(self):
        if hasattr(self, 'current_param_frame'):
            self.current_param_frame.destroy()
            self.current_param_frame = None
        
        model = self.model_var.get()
        kinetics = self.kinetics_var.get()
        chem = self.chem_method_var.get()
        self.current_param_frame = ttk.Frame(self.param_frame_container)
        self.current_param_frame.pack(fill=tk.BOTH, expand=True)

        # 滚动条
        canvas = tk.Canvas(self.current_param_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(self.current_param_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 标题
        ttk.Label(scrollable_frame, text=f"Parameters for {model} with {kinetics}", 
                 style='Header.TLabel').grid(row=0, column=0, columnspan=9, pady=5)

        # 公共参数
        ttk.Label(scrollable_frame, text="Temperature (T):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.temp_entry = ttk.Entry(scrollable_frame, width=12)
        self.temp_entry.grid(row=1, column=1, sticky=tk.W, pady=2)
        self.temp_entry.insert(0, "298.15")
        ttk.Label(scrollable_frame, text="K").grid(row=1, column=2, sticky=tk.W, padx=5)

        ttk.Label(scrollable_frame, text="ΔGw (eV):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        delta_gw_entry = ttk.Entry(scrollable_frame, width=12, textvariable=self.delta_gw_var)
        delta_gw_entry.grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Label(scrollable_frame, text="eV").grid(row=2, column=2, sticky=tk.W, padx=5)


    
        # 模型特定参数
        if model == "ER-AOM":
            if kinetics == "Butler-Volmer kinetics":
                self.create_er_aom_bv_parameters(scrollable_frame)
            elif kinetics == "Marcus kinetics":
                self.create_er_aom_marcus_parameters(scrollable_frame)
            else:
                self.create_er_aom_mg_parameters(scrollable_frame)

        else:
            if kinetics == "Butler-Volmer kinetics":
                self.create_lh_aom_bv_parameters(scrollable_frame)
            elif kinetics == "Marcus kinetics":
                self.create_lh_aom_marcus_parameters(scrollable_frame)
            else:
                self.create_lh_aom_mg_parameters(scrollable_frame)

        if kinetics == "Butler-Volmer kinetics":
            self.bv_sub_frame.pack(anchor=tk.W, pady=5)  # 显示子选项
        else:
            self.bv_sub_frame.pack_forget()  # 隐藏子选项 

    # 以下是完整的参数创建函数
    def create_er_aom_bv_parameters(self, frame):
        self.er_aom_bv_entries.clear()
        # Ea0
        ttk.Label(frame, text="Ea,0:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.ea0_entry = ttk.Entry(frame, width=12)
        self.ea0_entry.grid(row=3, column=1, sticky=tk.W, pady=2)
        self.ea0_entry.insert(0, "0.5")
        ttk.Label(frame, text="eV").grid(row=3, column=2, sticky=tk.W, padx=5)

        # Headers
        ttk.Label(frame, text="ΔG (eV)").grid(row=4, column=1, pady=5)
        ttk.Label(frame, text="γ").grid(row=4, column=3, pady=5)
        ttk.Label(frame, text="β").grid(row=4, column=5, pady=5)
        ttk.Label(frame, text="z").grid(row=4, column=7, pady=5)

        # ΔG, γ, β, z for steps 1-4
        self.er_aom_bv_entries = []
        for i, step in enumerate(range(1, 5), start=5):
            step_entries = {}
            
            ttk.Label(frame, text=f"Step {step}:").grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)
            
            # ΔG
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            entry.insert(0, "0.1")
            step_entries['deltaG'] = entry
            
            # γ
            gamma_value = "1.3863" if self.bv_method_var.get() == "Softplus" else "0.5"  
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=3, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, gamma_value)
            step_entries['gamma'] = entry
            
            # β
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=5, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "0.5")
            step_entries['beta'] = entry
            
            # z
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=7, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "1")
            step_entries['z'] = entry
            
            self.er_aom_bv_entries.append(step_entries)

    def create_er_aom_marcus_parameters(self, frame):
        self.er_aom_marcus_entries.clear()
        # Headers
        ttk.Label(frame, text="ΔG (eV)").grid(row=3, column=1, pady=5)
        ttk.Label(frame, text="λ (eV)").grid(row=3, column=3, pady=5)
        ttk.Label(frame, text="z").grid(row=3, column=5, pady=5)

        # ΔG, λ, z for steps 1-4
        self.er_aom_marcus_entries = []
        for i, step in enumerate(range(1, 5), start=4):
            step_entries = {}
            
            ttk.Label(frame, text=f"Step {step}:").grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)
            
            # ΔG
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            entry.insert(0, "0.1")
            step_entries['deltaG'] = entry
            
            # λ
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=3, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "2")
            step_entries['lambda'] = entry
            
            # z
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=5, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "1")
            step_entries['z'] = entry
            
            self.er_aom_marcus_entries.append(step_entries)

    def create_er_aom_mg_parameters(self, frame):
        self.er_aom_mg_entries.clear()
        # Headers
        ttk.Label(frame, text="ΔG (eV)").grid(row=3, column=1, pady=5)
        ttk.Label(frame, text="λ (eV)").grid(row=3, column=3, pady=5)
        ttk.Label(frame, text="z").grid(row=3, column=5, pady=5)

        # ΔG, λ, z for steps 1-4
        self.er_aom_mag_entries = []
        for i, step in enumerate(range(1, 5), start=4):
            step_entries = {}
            
            ttk.Label(frame, text=f"Step {step}:").grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)
            
            # ΔG
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            entry.insert(0, "0.1")
            step_entries['deltaG'] = entry
            
            # λ
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=3, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "2")
            step_entries['lambda'] = entry
            
            # z
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=5, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "1")
            step_entries['z'] = entry
            
            self.er_aom_mg_entries.append(step_entries)

    def create_lh_aom_bv_parameters(self, frame):
        self.lh_aom_bv_entries.clear()
        # Ea0
        ttk.Label(frame, text="Ea,0:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.ea0_entry = ttk.Entry(frame, width=12)
        self.ea0_entry.grid(row=3, column=1, sticky=tk.W, pady=2)
        self.ea0_entry.insert(0, "0.5")
        ttk.Label(frame, text="eV").grid(row=3, column=2, sticky=tk.W, padx=5)

        # Headers
        ttk.Label(frame, text="ΔG (eV)").grid(row=5, column=1, pady=5)
        ttk.Label(frame, text="γ").grid(row=5, column=3, pady=5)
        ttk.Label(frame, text="β").grid(row=5, column=5, pady=5)
        ttk.Label(frame, text="z").grid(row=5, column=7, pady=5)
        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=8, pady=10, sticky='ew')
        ttk.Label(frame, text="CHR", style='Header.TLabel').grid(row=4, column=0, columnspan=8, pady=5)
        ttk.Separator(frame, orient='horizontal').grid(row=12, column=0, columnspan=8, pady=10, sticky='ew')
        ttk.Label(frame, text="CR (Step 5)", style='Header.TLabel').grid(row=12, column=0, columnspan=8, pady=5)
            
        # Steps:1,21,22,31,32,4
        steps = [1,21,22,31,32,4]
        self.lh_aom_bv_entries = []
        for i, step_num in enumerate(steps, start=6):
            step_entries = {'step': step_num} 
            
            ttk.Label(frame, text=f"Step {step_num}:").grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)
            
            # ΔG
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            entry.insert(0, "0.1")
            step_entries['deltaG'] = entry
            
            # γ
            gamma_value = "1.3863" if self.bv_method_var.get() == "Softplus" else "0.5"  
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=3, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, gamma_value)
            step_entries['gamma'] = entry
            
            # β
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=5, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "0.5")
            step_entries['beta'] = entry
            
            # z
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=7, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "1")
            step_entries['z'] = entry

            self.lh_aom_bv_entries.append(step_entries)
        
        # Steps: 5
        steps = [5]
        for i, step_num in enumerate(steps, start=13):
            step_entries = {'step': step_num} 
            
            ttk.Label(frame, text=f"Step {step_num}:").grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)
            
            # ΔG
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            entry.insert(0, "-0.2")
            step_entries['deltaG'] = entry
            
            # γ
            gamma_value = "1.3863" if self.chem_method_var.get() == "Softplus" else "0.5"
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=3, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, gamma_value)
            step_entries['gamma'] = entry

            # β
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=5, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "0.5")
            step_entries['beta'] = entry
            
            # z
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=7, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "1")
            step_entries['z'] = entry
            
            self.lh_aom_bv_entries.append(step_entries)
            
    def create_lh_aom_marcus_parameters(self, frame):
        self.lh_aom_marcus_entries.clear()
        # Ea,0 参数
        ttk.Label(frame, text="Ea,0:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.ea0_entry = ttk.Entry(frame, width=12)
        self.ea0_entry.grid(row=3, column=1, sticky=tk.W, pady=2)
        self.ea0_entry.insert(0, "0.5")
        ttk.Label(frame, text="eV").grid(row=3, column=2, sticky=tk.W, padx=5)

        # Headers
        ttk.Label(frame, text="ΔG (eV)").grid(row=5, column=1, pady=5)
        ttk.Label(frame, text="λ (eV)").grid(row=5, column=3, pady=5)
        ttk.Label(frame, text="γ").grid(row=5, column=5, pady=5)
        ttk.Label(frame, text="z").grid(row=5, column=7, pady=5)
        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=8, pady=10, sticky='ew')
        ttk.Label(frame, text="CHR", style='Header.TLabel').grid(row=4, column=0, columnspan=8, pady=5)
        ttk.Separator(frame, orient='horizontal').grid(row=12, column=0, columnspan=8, pady=10, sticky='ew')
        ttk.Label(frame, text="CR (Step 5)", style='Header.TLabel').grid(row=12, column=0, columnspan=8, pady=5)
            
        # Steps: 1, 21, 22, 31, 32, 4
        steps = [1, 21, 22, 31, 32, 4]
        self.lh_aom_marcus_entries = []
        for i, step_num in enumerate(steps, start=6):
            step_entries = {'step': step_num} 
            
            ttk.Label(frame, text=f"Step {step_num}:").grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)

            # ΔG
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            entry.insert(0, "0.1")
            step_entries['deltaG'] = entry
            
            # λ
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=3, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "2")
            step_entries['lambda'] = entry

            # γ
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=5, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "0.5")
            step_entries['gamma'] = entry
        
            # z
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=7, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "1")
            step_entries['z'] = entry

            self.lh_aom_marcus_entries.append(step_entries)

        # Steps: 5
        steps = [5]
        for i, step_num in enumerate(steps, start=13):
            step_entries = {'step': step_num} 
            
            ttk.Label(frame, text=f"Step {step_num}:").grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)
            
            # ΔG
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            entry.insert(0, "-0.2")
            step_entries['deltaG'] = entry
            
            # λ
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=3, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "2")
            step_entries['lambda'] = entry

            # γ
            gamma_value = "1.3863" if self.chem_method_var.get() == "Softplus" else "0.5"
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=5, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, gamma_value)
            step_entries['gamma'] = entry
        
            # z
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=7, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "1")
            step_entries['z'] = entry
            
            self.lh_aom_marcus_entries.append(step_entries)

    def create_lh_aom_mg_parameters(self, frame):
        self.lh_aom_mg_entries.clear()
        # Ea,0 参数
        ttk.Label(frame, text="Ea,0:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.ea0_entry = ttk.Entry(frame, width=12)
        self.ea0_entry.grid(row=3, column=1, sticky=tk.W, pady=2)
        self.ea0_entry.insert(0, "0.5")
        ttk.Label(frame, text="eV").grid(row=3, column=2, sticky=tk.W, padx=5)

        # Headers
        ttk.Label(frame, text="ΔG (eV)").grid(row=5, column=1, pady=5)
        ttk.Label(frame, text="λ (eV)").grid(row=5, column=3, pady=5)
        ttk.Label(frame, text="γ").grid(row=5, column=5, pady=5)
        ttk.Label(frame, text="z").grid(row=5, column=7, pady=5)
        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=8, pady=10, sticky='ew')
        ttk.Label(frame, text="CHR", style='Header.TLabel').grid(row=4, column=0, columnspan=8, pady=5)
        ttk.Separator(frame, orient='horizontal').grid(row=12, column=0, columnspan=8, pady=10, sticky='ew')
        ttk.Label(frame, text="CR (Step 5)", style='Header.TLabel').grid(row=12, column=0, columnspan=8, pady=5)
            

        # Steps: 1, 21, 22, 31, 32, 4
        steps = [1, 21, 22, 31, 32, 4]
        self.lh_aom_marcus_entries = []
        for i, step_num in enumerate(steps, start=6):
            step_entries = {'step': step_num} 
            
            ttk.Label(frame, text=f"Step {step_num}:").grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)
            
            # ΔG
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            entry.insert(0, "0.1")
            step_entries['deltaG'] = entry
            
            # λ
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=3, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "2")
            step_entries['lambda'] = entry

            # γ
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=5, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "0.5")
            step_entries['gamma'] = entry
        
            # z
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=7, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "1")
            step_entries['z'] = entry
            
            self.lh_aom_mg_entries.append(step_entries)

        # Steps: 5
        steps = [5]
        for i, step_num in enumerate(steps, start=13):
            step_entries = {'step': step_num} 
            
            ttk.Label(frame, text=f"Step {step_num}:").grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)
            
            # ΔG
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            entry.insert(0, "-0.2")
            step_entries['deltaG'] = entry
            
            # λ
            entry = ttk.Entry(frame, width=12)
            entry.grid(row=i, column=3, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "2")
            step_entries['lambda'] = entry

            # γ
            gamma_value = "1.3863" if self.chem_method_var.get() == "Softplus" else "0.5"  
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=5, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, gamma_value)
            step_entries['gamma'] = entry
        
            # z
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i, column=7, sticky=tk.W, pady=2, padx=5)
            entry.insert(0, "1")
            step_entries['z'] = entry
            
            self.lh_aom_mg_entries.append(step_entries)

    # 以下是完整的计算函数
    
    def calculate(self):
        try:
            # Get temperature
            T = float(self.temp_entry.get())
        
            scan_mode = self.variable_var.get()
        
            if scan_mode == "2D":
                # 2D扫描逻辑
                eta_values = np.arange(
                    self.eta_start_var.get(),
                    self.eta_end_var.get() + self.eta_step_var.get()/2,
                    self.eta_step_var.get()
                )
                ph_values = np.arange(
                    self.ph_start_var.get(),
                    self.ph_end_var.get() + self.ph_step_var.get()/2,
                    self.ph_step_var.get()
                )
                eta_grid, ph_grid = np.meshgrid(eta_values, ph_values)
            
                # 获取模型和动力学参数
                model = self.model_var.get()
                kinetics = self.kinetics_var.get()
            
                # 初始化结果矩阵
                Z_lgr = np.zeros_like(eta_grid)
                Z_theta = np.zeros_like(eta_grid)
            
                # 根据模型获取参数
                if model == "LH-AOM":
                    if kinetics == "Butler-Volmer kinetics":
                        ea0 = float(self.ea0_entry.get())
                        steps = {}
                        for step_entry in self.lh_aom_bv_entries:
                            step_num = step_entry['step']
                            steps[step_num] = {
                                'deltaG': float(step_entry['deltaG'].get()),
                                'gamma': float(step_entry['gamma'].get()),
                                'beta': float(step_entry['beta'].get()),
                                'z': float(step_entry['z'].get())
                            }

                        # 遍历每个点计算
                        for i in range(len(ph_values)):
                            for j in range(len(eta_values)):
                                eta = eta_values[j]
                                ph = ph_values[i]
                                try:
                                    # 计算k值
                                    k = {}
                                    # Step 1
                                    step = steps[1]
                                    k['k1a'] = self.calculate_bv_ka(step, ea0, T, eta, ph)
                                    k['k-1a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, ph)
                                    k['k1b'] = self.calculate_bv_kb(step, ea0, T, eta, ph)
                                    k['k-1b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, ph)
                                
                                    # Step 21
                                    step = steps[21]
                                    k['k21a'] = self.calculate_bv_ka(step, ea0, T, eta, ph)
                                    k['k-21a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, ph)
                                    k['k21b'] = self.calculate_bv_kb(step, ea0, T, eta, ph)
                                    k['k-21b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, ph)
                                
                                    # Step 22
                                    step = steps[22]
                                    k['k22a'] = self.calculate_bv_ka(step, ea0, T, eta, ph)
                                    k['k-22a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, ph)
                                    k['k22b'] = self.calculate_bv_kb(step, ea0, T, eta, ph)
                                    k['k-22b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, ph)
                                
                                    # Step 31
                                    step = steps[31]
                                    k['k31a'] = self.calculate_bv_ka(step, ea0, T, eta, ph)
                                    k['k-31a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, ph)
                                    k['k31b'] = self.calculate_bv_kb(step, ea0, T, eta, ph)
                                    k['k-31b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, ph)
                                
                                    # Step 32
                                    step = steps[32]
                                    k['k32a'] = self.calculate_bv_ka(step, ea0, T, eta, ph)
                                    k['k-32a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, ph)
                                    k['k32b'] = self.calculate_bv_kb(step, ea0, T, eta, ph)
                                    k['k-32b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, ph)
                                
                                    # Step 4
                                    step = steps[4]
                                    k['k4a'] = self.calculate_bv_ka(step, ea0, T, eta, ph)
                                    k['k-4a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, ph)
                                    k['k4b'] = self.calculate_bv_kb(step, ea0, T, eta, ph)
                                    k['k-4b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, ph)
                                
                                    # Step 5
                                    step = steps[5]
                                    gamma = step['gamma']
                                    deltaG = step['deltaG']
                                    if self.chem_method_var.get() == "BEP":
                                        k['k5'] = (kB * T / h) * math.exp(-(ea0 + step['gamma'] * step['deltaG']) / (kB * T))
                                        k['k-5'] = (kB * T / h) * math.exp(-(ea0 - step['gamma'] * step['deltaG']) / (kB * T))
                                    else:
                                        Ea5 = (1/gamma) * math.log(1 + (math.exp(gamma * deltaG)))
                                        Ea_minus5 = (1/gamma) * math.log(1 + (math.exp(-gamma * deltaG)))
                                        k['k5'] = (kB * T / h) * math.exp(-Ea5 / (kB * T))
                                        k['k-5'] = (kB * T / h) * math.exp(-Ea_minus5 / (kB * T))

                                    # 组合k值
                                    k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - ph)
                                    k['k-1'] = k['k-1a'] * 10**-ph + k['k-1b']
                                    k['k21'] = k['k21a'] + k['k21b'] * 10**-(14 - ph)
                                    k['k-21'] = k['k-21a'] * 10**-ph + k['k-21b']
                                    k['k22'] = k['k22a'] + k['k22b'] * 10**-(14 - ph)
                                    k['k-22'] = k['k-22a'] * 10**-ph + k['k-22b']
                                    k['k31'] = k['k31a'] + k['k31b'] * 10**-(14 - ph)
                                    k['k-31'] = k['k-31a'] * 10**-ph + k['k-31b']
                                    k['k32'] = k['k32a'] + k['k32b'] * 10**-(14 - ph)
                                    k['k-32'] = k['k-32a'] * 10**-ph + k['k-32b']
                                    k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - ph)
                                    k['k-4'] = k['k-4a'] * 10**-ph + k['k-4b']
                                
                                    # 计算θ
                                    theta = self.calculate_lh_aom_theta(k)
                                
                                    # 计算r值
                                    r = {
                                        'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                        'r21': k['k21'] * theta['theta*OH'] - k['k-21'] * theta['theta*(OH)2'],
                                        'r22': k['k22'] * theta['theta*OH'] - k['k-22'] * theta['theta*O'],
                                        'r31': k['k31'] * theta['theta*(OH)2'] - k['k-31'] * theta['theta*O(OH)'],
                                        'r32': k['k32'] * theta['theta*O'] - k['k-32'] * theta['theta*O(OH)'],
                                        'r4': k['k4'] * theta['theta*O(OH)'] - k['k-4'] * theta['theta*O(O)'],
                                        'r5': k['k5'] * theta['theta*O(O)'] - k['k-5'] * theta['theta*']
                                    }
                                    r['r2'] = r['r21'] + r['r22']
                                    r['r3'] = r['r31'] + r['r32']
                        
                                
                                    # 记录结果
                                    current_r = r['r5']
                                    Z_lgr[i,j] = math.log10(abs(current_r)) if current_r != 0 else -np.inf
                                    Z_theta[i,j] = theta['theta*']
                                    
                                except Exception as e:
                                    Z_lgr[i,j] = np.nan
                                    Z_theta[i,j] = np.nan
                                    print(f"Error at eta={eta}, ph={ph}: {str(e)}")

                    elif kinetics == "Marcus kinetics":
                        ea0 = float(self.ea0_entry.get())
                        steps = {}
                        for step_entry in self.lh_aom_marcus_entries:
                            step_num = step_entry['step']
                            steps[step_num] = {
                                'deltaG': float(step_entry['deltaG'].get()),
                                'lambda': float(step_entry['lambda'].get()),
                                'gamma': float(step_entry['gamma'].get()),
                                'z': float(step_entry['z'].get())
                            }

                        # 遍历每个点计算
                        for i in range(len(ph_values)):
                            for j in range(len(eta_values)):
                                eta = eta_values[j]
                                ph = ph_values[i]
                                try:
                                    # 计算k值
                                    k = {}
                                    # Step 1
                                    step = steps[1]
                                    k['k1a'] = self.calculate_marcus_ka(step, T, eta, ph)
                                    k['k-1a'] = self.calculate_marcus_k_minus_a(step, T, eta, ph)
                                    k['k1b'] = self.calculate_marcus_kb(step, T, eta, ph)
                                    k['k-1b'] = self.calculate_marcus_k_minus_b(step, T, eta, ph)
                                
                                    # Step 21
                                    step = steps[21]
                                    k['k21a'] = self.calculate_marcus_ka(step, T, eta, ph)
                                    k['k-21a'] = self.calculate_marcus_k_minus_a(step, T, eta, ph)
                                    k['k21b'] = self.calculate_marcus_kb(step, T, eta, ph)
                                    k['k-21b'] = self.calculate_marcus_k_minus_b(step, T, eta, ph)
                                
                                    # Step 22
                                    step = steps[22]
                                    k['k22a'] = self.calculate_marcus_ka(step, T, eta, ph)
                                    k['k-22a'] = self.calculate_marcus_k_minus_a(step, T, eta, ph)
                                    k['k22b'] = self.calculate_marcus_kb(step, T, eta, ph)
                                    k['k-22b'] = self.calculate_marcus_k_minus_b(step, T, eta, ph)
                                
                                    # Step 31
                                    step = steps[31]
                                    k['k31a'] = self.calculate_marcus_ka(step, T, eta, ph)
                                    k['k-31a'] = self.calculate_marcus_k_minus_a(step, T, eta, ph)
                                    k['k31b'] = self.calculate_marcus_kb(step, T, eta, ph)
                                    k['k-31b'] = self.calculate_marcus_k_minus_b(step, T, eta, ph)
                                
                                    # Step 32
                                    step = steps[32]
                                    k['k32a'] = self.calculate_marcus_ka(step, T, eta, ph)
                                    k['k-32a'] = self.calculate_marcus_k_minus_a(step, T, eta, ph)
                                    k['k32b'] = self.calculate_marcus_kb(step, T, eta, ph)
                                    k['k-32b'] = self.calculate_marcus_k_minus_b(step, T, eta, ph)
                                
                                    # Step 4
                                    step = steps[4]
                                    k['k4a'] = self.calculate_marcus_ka(step, T, eta, ph)
                                    k['k-4a'] = self.calculate_marcus_k_minus_a(step, T, eta, ph)
                                    k['k4b'] = self.calculate_marcus_kb(step, T, eta, ph)
                                    k['k-4b'] = self.calculate_marcus_k_minus_b(step, T, eta, ph)
                                
                                    # Step 5
                                    step = steps[5]
                                    gamma = step['gamma']
                                    deltaG = step['deltaG']
                                    if self.chem_method_var.get() == "BEP":
                                        k['k5'] = (kB * T / h) * math.exp(-(ea0 + step['gamma'] * step['deltaG']) / (kB * T))
                                        k['k-5'] = (kB * T / h) * math.exp(-(ea0 - step['gamma'] * step['deltaG']) / (kB * T))
                                    else:
                                        Ea5 = (1/gamma) * math.log(1 + (math.exp(gamma * deltaG)))
                                        Ea_minus5 = (1/gamma) * math.log(1 + (math.exp(-gamma * deltaG)))
                                        k['k5'] = (kB * T / h) * math.exp(-Ea5 / (kB * T))
                                        k['k-5'] = (kB * T / h) * math.exp(-Ea_minus5 / (kB * T))

                                    # 组合k值
                                    k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - ph)
                                    k['k-1'] = k['k-1a'] * 10**-ph + k['k-1b']
                                    k['k21'] = k['k21a'] + k['k21b'] * 10**-(14 - ph)
                                    k['k-21'] = k['k-21a'] * 10**-ph + k['k-21b']
                                    k['k22'] = k['k22a'] + k['k22b'] * 10**-(14 - ph)
                                    k['k-22'] = k['k-22a'] * 10**-ph + k['k-22b']
                                    k['k31'] = k['k31a'] + k['k31b'] * 10**-(14 - ph)
                                    k['k-31'] = k['k-31a'] * 10**-ph + k['k-31b']
                                    k['k32'] = k['k32a'] + k['k32b'] * 10**-(14 - ph)
                                    k['k-32'] = k['k-32a'] * 10**-ph + k['k-32b']
                                    k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - ph)
                                    k['k-4'] = k['k-4a'] * 10**-ph + k['k-4b']
                                
                                    # 计算θ
                                    theta = self.calculate_lh_aom_theta(k)
                                
                                    # 计算r值
                                    r = {
                                        'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                        'r21': k['k21'] * theta['theta*OH'] - k['k-21'] * theta['theta*(OH)2'],
                                        'r22': k['k22'] * theta['theta*OH'] - k['k-22'] * theta['theta*O'],
                                        'r31': k['k31'] * theta['theta*(OH)2'] - k['k-31'] * theta['theta*O(OH)'],
                                        'r32': k['k32'] * theta['theta*O'] - k['k-32'] * theta['theta*O(OH)'],
                                        'r4': k['k4'] * theta['theta*O(OH)'] - k['k-4'] * theta['theta*O(O)'],
                                        'r5': k['k5'] * theta['theta*O(O)'] - k['k-5'] * theta['theta*']
                                    }
                                    r['r2'] = r['r21'] + r['r22']
                                    r['r3'] = r['r31'] + r['r32']
                        
                                
                                    # 记录结果
                                    current_r = r['r5']
                                    Z_lgr[i,j] = math.log10(abs(current_r)) if current_r != 0 else -np.inf
                                    Z_theta[i,j] = theta['theta*']
                                    
                                except Exception as e:
                                    Z_lgr[i,j] = np.nan
                                    Z_theta[i,j] = np.nan
                                    print(f"Error at eta={eta}, ph={ph}: {str(e)}")

                    elif kinetics == "Marcus-Gerischer kinetics":
                        ea0 = float(self.ea0_entry.get())
                        steps = {}
                        for step_entry in self.lh_aom_mg_entries:
                            step_num = step_entry['step']
                            steps[step_num] = {
                                'deltaG': float(step_entry['deltaG'].get()),
                                'lambda': float(step_entry['lambda'].get()),
                                'gamma': float(step_entry['gamma'].get()),
                                'z': float(step_entry['z'].get())
                            }

                        # 遍历每个点计算
                        for i in range(len(ph_values)):
                            for j in range(len(eta_values)):
                                eta = eta_values[j]
                                ph = ph_values[i]
                                try:
                                    # 计算k值
                                    k = {}
                                    # Step 1
                                    step = steps[1]
                                    k['k1a'] = self.calculate_mg_ka(step, T, eta, ph)
                                    k['k-1a'] = self.calculate_mg_k_minus_a(step, T, eta, ph)
                                    k['k1b'] = self.calculate_mg_kb(step, T, eta, ph)
                                    k['k-1b'] = self.calculate_mg_k_minus_b(step, T, eta, ph)
                                
                                    # Step 21
                                    step = steps[21]
                                    k['k21a'] = self.calculate_mg_ka(step, T, eta, ph)
                                    k['k-21a'] = self.calculate_mg_k_minus_a(step, T, eta, ph)
                                    k['k21b'] = self.calculate_mg_kb(step, T, eta, ph)
                                    k['k-21b'] = self.calculate_mg_k_minus_b(step, T, eta, ph)
                                
                                    # Step 22
                                    step = steps[22]
                                    k['k22a'] = self.calculate_mg_ka(step, T, eta, ph)
                                    k['k-22a'] = self.calculate_mg_k_minus_a(step, T, eta, ph)
                                    k['k22b'] = self.calculate_mg_kb(step, T, eta, ph)
                                    k['k-22b'] = self.calculate_mg_k_minus_b(step, T, eta, ph)
                                
                                    # Step 31
                                    step = steps[31]
                                    k['k31a'] = self.calculate_mg_ka(step, T, eta, ph)
                                    k['k-31a'] = self.calculate_mg_k_minus_a(step, T, eta, ph)
                                    k['k31b'] = self.calculate_mg_kb(step, T, eta, ph)
                                    k['k-31b'] = self.calculate_mg_k_minus_b(step, T, eta, ph)
                                
                                    # Step 32
                                    step = steps[32]
                                    k['k32a'] = self.calculate_mg_ka(step, T, eta, ph)
                                    k['k-32a'] = self.calculate_mg_k_minus_a(step, T, eta, ph)
                                    k['k32b'] = self.calculate_mg_kb(step, T, eta, ph)
                                    k['k-32b'] = self.calculate_mg_k_minus_b(step, T, eta, ph)
                                
                                    # Step 4
                                    step = steps[4]
                                    k['k4a'] = self.calculate_mg_ka(step, T, eta, ph)
                                    k['k-4a'] = self.calculate_mg_k_minus_a(step, T, eta, ph)
                                    k['k4b'] = self.calculate_mg_kb(step, T, eta, ph)
                                    k['k-4b'] = self.calculate_mg_k_minus_b(step, T, eta, ph)
                                
                                    # Step 5
                                    step = steps[5]
                                    gamma = step['gamma']
                                    deltaG = step['deltaG']
                                    if self.chem_method_var.get() == "BEP":
                                        k['k5'] = (kB * T / h) * math.exp(-(ea0 + step['gamma'] * step['deltaG']) / (kB * T))
                                        k['k-5'] = (kB * T / h) * math.exp(-(ea0 - step['gamma'] * step['deltaG']) / (kB * T))
                                    else:
                                        Ea5 = (1/gamma) * math.log(1 + (math.exp(gamma * deltaG)))
                                        Ea_minus5 = (1/gamma) * math.log(1 + (math.exp(-gamma * deltaG)))
                                        k['k5'] = (kB * T / h) * math.exp(-Ea5 / (kB * T))
                                        k['k-5'] = (kB * T / h) * math.exp(-Ea_minus5 / (kB * T))

                                    # 组合k值
                                    k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - ph)
                                    k['k-1'] = k['k-1a'] * 10**-ph + k['k-1b']
                                    k['k21'] = k['k21a'] + k['k21b'] * 10**-(14 - ph)
                                    k['k-21'] = k['k-21a'] * 10**-ph + k['k-21b']
                                    k['k22'] = k['k22a'] + k['k22b'] * 10**-(14 - ph)
                                    k['k-22'] = k['k-22a'] * 10**-ph + k['k-22b']
                                    k['k31'] = k['k31a'] + k['k31b'] * 10**-(14 - ph)
                                    k['k-31'] = k['k-31a'] * 10**-ph + k['k-31b']
                                    k['k32'] = k['k32a'] + k['k32b'] * 10**-(14 - ph)
                                    k['k-32'] = k['k-32a'] * 10**-ph + k['k-32b']
                                    k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - ph)
                                    k['k-4'] = k['k-4a'] * 10**-ph + k['k-4b']
                                
                                    # 计算θ
                                    theta = self.calculate_lh_aom_theta(k)
                                
                                    # 计算r值
                                    r = {
                                        'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                        'r21': k['k21'] * theta['theta*OH'] - k['k-21'] * theta['theta*(OH)2'],
                                        'r22': k['k22'] * theta['theta*OH'] - k['k-22'] * theta['theta*O'],
                                        'r31': k['k31'] * theta['theta*(OH)2'] - k['k-31'] * theta['theta*O(OH)'],
                                        'r32': k['k32'] * theta['theta*O'] - k['k-32'] * theta['theta*O(OH)'],
                                        'r4': k['k4'] * theta['theta*O(OH)'] - k['k-4'] * theta['theta*O(O)'],
                                        'r5': k['k5'] * theta['theta*O(O)'] - k['k-5'] * theta['theta*']
                                    }
                                    r['r2'] = r['r21'] + r['r22']
                                    r['r3'] = r['r31'] + r['r32']
                        
                                
                                    # 记录结果
                                    current_r = r['r5']
                                    Z_lgr[i,j] = math.log10(abs(current_r)) if current_r != 0 else -np.inf
                                    Z_theta[i,j] = theta['theta*']


                                except Exception as e:
                                    Z_lgr[i,j] = np.nan
                                    Z_theta[i,j] = np.nan
                                    print(f"Error at eta={eta}, ph={ph}: {str(e)}")                        

                elif model == "ER-AOM":
                    if kinetics == "Butler-Volmer kinetics":
                        ea0 = float(self.ea0_entry.get())
                        steps = []
                        for step_entry in self.er_aom_bv_entries:
                            steps.append({
                                'deltaG': float(step_entry['deltaG'].get()),
                                'gamma': float(step_entry['gamma'].get()),
                                'beta': float(step_entry['beta'].get()),
                                'z': float(step_entry['z'].get())
                            })
                    
                        # 遍历每个点计算
                        for i in range(len(ph_values)):
                            for j in range(len(eta_values)):
                                eta = eta_values[j]
                                ph = ph_values[i]
                                try:
                                    # 计算k值
                                    k = {} 
                                    for step_idx, step in enumerate(steps, start=1):
                                        # Forward reaction (a)
                                        k[f'k{step_idx}a'] = self.calculate_bv_ka(step, ea0, T, eta, ph)
                                        # Backward reaction (-a)
                                        k[f'k-{step_idx}a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, ph)
                                        # Forward reaction (b)
                                        k[f'k{step_idx}b'] = self.calculate_bv_kb(step, ea0, T, eta, ph)
                                        # Backward reaction (-b)
                                        k[f'k-{step_idx}b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, ph)
                                
                                    # 组合k值
                                    k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - ph)
                                    k['k-1'] = k['k-1a'] * 10**-ph + k['k-1b']
                                    k['k2'] = k['k2a'] + k['k2b'] * 10**-(14 - ph)
                                    k['k-2'] = k['k-2a'] * 10**-ph + k['k-2b']
                                    k['k3'] = k['k3a'] + k['k3b'] * 10**-(14 - ph)
                                    k['k-3'] = k['k-3a'] * 10**-ph + k['k-3b']
                                    k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - ph)
                                    k['k-4'] = k['k-4a'] * 10**-ph + k['k-4b']
                                
                                    # 计算θ
                                    theta = self.calculate_er_aom_theta(k)
                                
                                    # 计算r值
                                    r = {
                                        'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                        'r2': k['k2'] * theta['theta*OH'] - k['k-2'] * theta['theta*O'],
                                        'r3': k['k3'] * theta['theta*O'] - k['k-3'] * theta['theta*OOH'],
                                        'r4': k['k4'] * theta['theta*OOH'] - k['k-4'] * theta['theta*']
                                    }
                                
                                    # 记录结果
                                    current_r = r['r4']  # 根据实际需要调整
                                    Z_lgr[i,j] = math.log10(abs(current_r)) if current_r != 0 else -np.inf
                                    Z_theta[i,j] = theta['theta*']
                                    
                                except Exception as e:
                                    Z_lgr[i,j] = np.nan
                                    Z_theta[i,j] = np.nan
                                    print(f"Error at eta={eta}, ph={ph}: {str(e)}")

                    elif kinetics == "Marcus kinetics":
                        steps = []
                        for step_entry in self.er_aom_marcus_entries:
                            steps.append({
                                'deltaG': float(step_entry['deltaG'].get()),
                                'lambda': float(step_entry['lambda'].get()),
                                'z': float(step_entry['z'].get())
                            })
                    
                        # 遍历每个点计算
                        for i in range(len(ph_values)):
                            for j in range(len(eta_values)):
                                eta = eta_values[j]
                                ph = ph_values[i]
                                try:
                                    # 计算k值
                                    k = {} 
                                    for step_idx, step in enumerate(steps, start=1):
                                        # Forward reaction (a)
                                        k[f'k{step_idx}a'] = self.calculate_marcus_ka(step, T, eta, ph)
                                        # Backward reaction (-a)
                                        k[f'k-{step_idx}a'] = self.calculate_marcus_k_minus_a(step, T, eta, ph)
                                        # Forward reaction (b)
                                        k[f'k{step_idx}b'] = self.calculate_marcus_kb(step, T, eta, ph)
                                        # Backward reaction (-b)
                                        k[f'k-{step_idx}b'] = self.calculate_marcus_k_minus_b(step, T, eta, ph)
                                
                                    # 组合k值
                                    k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - ph)
                                    k['k-1'] = k['k-1a'] * 10**-ph + k['k-1b']
                                    k['k2'] = k['k2a'] + k['k2b'] * 10**-(14 - ph)
                                    k['k-2'] = k['k-2a'] * 10**-ph + k['k-2b']
                                    k['k3'] = k['k3a'] + k['k3b'] * 10**-(14 - ph)
                                    k['k-3'] = k['k-3a'] * 10**-ph + k['k-3b']
                                    k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - ph)
                                    k['k-4'] = k['k-4a'] * 10**-ph + k['k-4b']
                                
                                    # 计算θ
                                    theta = self.calculate_er_aom_theta(k)
                                
                                    # 计算r值
                                    r = {
                                        'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                        'r2': k['k2'] * theta['theta*OH'] - k['k-2'] * theta['theta*O'],
                                        'r3': k['k3'] * theta['theta*O'] - k['k-3'] * theta['theta*OOH'],
                                        'r4': k['k4'] * theta['theta*OOH'] - k['k-4'] * theta['theta*']
                                    }
                                
                                    # 记录结果
                                    current_r = r['r4']  # 根据实际需要调整
                                    Z_lgr[i,j] = math.log10(abs(current_r)) if current_r != 0 else -np.inf
                                    Z_theta[i,j] = theta['theta*']

                                except Exception as e:
                                    Z_lgr[i,j] = np.nan
                                    Z_theta[i,j] = np.nan
                                    print(f"Error at eta={eta}, ph={ph}: {str(e)}")
    
                    elif kinetics == "Marcus-Gerischer kinetics":
                        steps = []
                        for step_entry in self.er_aom_mg_entries:
                            steps.append({
                                'deltaG': float(step_entry['deltaG'].get()),
                                'lambda': float(step_entry['lambda'].get()),
                                'z': float(step_entry['z'].get())
                            })
        
                        # 遍历每个点计算
                        for i in range(len(ph_values)):
                            for j in range(len(eta_values)):
                                eta = eta_values[j]
                                ph = ph_values[i]
                                try:
                                    # 计算k值
                                    k = {} 
                                    for step_idx, step in enumerate(steps, start=1):
                                        # Forward reaction (a)
                                        k[f'k{step_idx}a'] = self.calculate_mg_ka(step, T, eta, ph)
                                        # Backward reaction (-a)
                                        k[f'k-{step_idx}a'] = self.calculate_mg_k_minus_a(step, T, eta, ph)
                                        # Forward reaction (b)
                                        k[f'k{step_idx}b'] = self.calculate_mg_kb(step, T, eta, ph)
                                        # Backward reaction (-b)
                                        k[f'k-{step_idx}b'] = self.calculate_mg_k_minus_b(step, T, eta, ph)
                                
                                    # 组合k值
                                    k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - ph)
                                    k['k-1'] = k['k-1a'] * 10**-ph + k['k-1b']
                                    k['k2'] = k['k2a'] + k['k2b'] * 10**-(14 - ph)
                                    k['k-2'] = k['k-2a'] * 10**-ph + k['k-2b']
                                    k['k3'] = k['k3a'] + k['k3b'] * 10**-(14 - ph)
                                    k['k-3'] = k['k-3a'] * 10**-ph + k['k-3b']
                                    k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - ph)
                                    k['k-4'] = k['k-4a'] * 10**-ph + k['k-4b']
                                
                                    # 计算θ
                                    theta = self.calculate_er_aom_theta(k)
                                
                                    # 计算r值
                                    r = {
                                        'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                        'r2': k['k2'] * theta['theta*OH'] - k['k-2'] * theta['theta*O'],
                                        'r3': k['k3'] * theta['theta*O'] - k['k-3'] * theta['theta*OOH'],
                                        'r4': k['k4'] * theta['theta*OOH'] - k['k-4'] * theta['theta*']
                                    }
                                
                                    # 记录结果
                                    current_r = r['r4']  # 根据实际需要调整
                                    Z_lgr[i,j] = math.log10(abs(current_r)) if current_r != 0 else -np.inf
                                    Z_theta[i,j] = theta['theta*']
                                except Exception as e:
                                    Z_lgr[i,j] = np.nan
                                    Z_theta[i,j] = np.nan
                                    print(f"Error at eta={eta}, ph={ph}: {str(e)}")                                    
                                                             

                # 保存结果
                self.results_2d = {
                    'eta': eta_grid,
                    'ph': ph_grid,
                    'lgr': Z_lgr,
                    'theta': Z_theta
                }
                
                # 绘制等值线图
                self.create_contour_plot()
                messagebox.showinfo("计算完成", "二维扫描计算成功完成！")
                return
            
            else:
                # 原有的一维逻辑
                # Get variable range
                var_range = np.arange(self.start_var.get(),
                                    self.end_var.get() + self.step_var.get()/2,
                                    self.step_var.get())

                # Get fixed values based on selected variable
                if self.variable_var.get() == "η":
                    variable = var_range
                    fixed_value = self.fixed_ph_var.get()
                    fixed_label = "Fixed pH"
                else:
                    variable = var_range
                    fixed_value = self.fixed_eta_var.get()
                    fixed_label = "Fixed η"

                # Get model and kinetics type
                model = self.model_var.get()
                kinetics = self.kinetics_var.get()

                # Initialize results dictionary
                results = {
                    self.variable_var.get(): variable,
                    fixed_label: [fixed_value] * len(variable)
                }

                # Perform calculations based on model and kinetics
                # ... 原有的一维计算代码 ...
                if model == "ER-AOM":
                    if kinetics == "Butler-Volmer kinetics":
                        # Get parameters for ER-AOM with Butler-Volmer kinetics
                        ea0 = float(self.ea0_entry.get())                              
                    
                        # Get step parameters
                        steps = []
                        for step_entry in self.er_aom_bv_entries:
                            step = {
                                'deltaG': float(step_entry['deltaG'].get()),
                                'gamma': float(step_entry['gamma'].get()),
                                'beta': float(step_entry['beta'].get()),
                                'z': float(step_entry['z'].get())
                            }
                            steps.append(step)
                    
                        # Calculate for each variable value
                        k_values = []
                        theta_values = []
                        r_values = []
                    
                        for var in variable:
                            if self.variable_var.get() == "η":
                                eta = var
                                pH = fixed_value
                            else:
                                eta = fixed_value
                                pH = var
                        
                            # Calculate all k values
                            k = {}
                            for i, step in enumerate(steps, start=1):
                                # Forward reaction (a)
                                k[f'k{i}a'] = self.calculate_bv_ka(step, ea0, T, eta, pH)
                                # Backward reaction (-a)
                                k[f'k-{i}a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, pH)
                                # Forward reaction (b)
                                k[f'k{i}b'] = self.calculate_bv_kb(step, ea0, T, eta, pH)
                                # Backward reaction (-b)
                                k[f'k-{i}b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, pH)
                        
                            # Calculate combined k values
                            k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - pH)
                            k['k-1'] = k['k-1a'] * 10**-pH + k['k-1b']
                            k['k2'] = k['k2a'] + k['k2b'] * 10**-(14 - pH)
                            k['k-2'] = k['k-2a'] * 10**-pH + k['k-2b']
                            k['k3'] = k['k3a'] + k['k3b'] * 10**-(14 - pH)
                            k['k-3'] = k['k-3a'] * 10**-pH + k['k-3b']
                            k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - pH)
                            k['k-4'] = k['k-4a'] * 10**-pH + k['k-4b']
                        
                            # Calculate θ values
                            theta = self.calculate_er_aom_theta(k)
                        
                            # Calculate r values
                            r = {
                                'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                'r2': k['k2'] * theta['theta*OH'] - k['k-2'] * theta['theta*O'],
                                'r3': k['k3'] * theta['theta*O'] - k['k-3'] * theta['theta*OOH'],
                                'r4': k['k4'] * theta['theta*OOH'] - k['k-4'] * theta['theta*']
                            }
                        
                            # Store results
                            k_values.append(k)
                            theta_values.append(theta)
                            r_values.append(r)
                    
                        # Prepare results for output
                        for i in range(1, 5):
                            results[f'k{i}'] = [k[f'k{i}'] for k in k_values]
                            results[f'k-{i}'] = [k[f'k-{i}'] for k in k_values]
                            results[f'lg(r{i})'] = [math.log10(abs(r[f'r{i}'])) if r[f'r{i}'] != 0 else -float('inf') for r in r_values]
                    
                        for theta_name in ['theta*', 'theta*OH', 'theta*O', 'theta*OOH']:
                            results[theta_name] = [theta[theta_name] for theta in theta_values]
                    
                    elif kinetics == "Marcus kinetics":
                        # Get step parameters
                        steps = []
                        for step_entry in self.er_aom_marcus_entries:
                            step = {
                                'deltaG': float(step_entry['deltaG'].get()),
                                'lambda': float(step_entry['lambda'].get()),
                                'z': float(step_entry['z'].get())
                            }
                            steps.append(step)
                    
                        # Calculate for each variable value
                        k_values = []
                        theta_values = []
                        r_values = []
                    
                        for var in variable:
                            if self.variable_var.get() == "η":
                                eta = var
                                pH = fixed_value
                            else:
                                eta = fixed_value
                                pH = var
                        
                            # Calculate all k values
                            k = {}
                            for i, step in enumerate(steps, start=1):
                                #"Marcus kinetics"
                                # Forward reaction (a)
                                k[f'k{i}a'] = self.calculate_marcus_ka(step, T, eta, pH)
                                # Backward reaction (-a)
                                k[f'k-{i}a'] = self.calculate_marcus_k_minus_a(step, T, eta, pH)
                                # Forward reaction (b)
                                k[f'k{i}b'] = self.calculate_marcus_kb(step, T, eta, pH)
                                # Backward reaction (-b)
                                k[f'k-{i}b'] = self.calculate_marcus_k_minus_b(step, T, eta, pH)
                        
                            # Calculate combined k values
                            k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - pH)
                            k['k-1'] = k['k-1a'] * 10**-pH + k['k-1b']
                            k['k2'] = k['k2a'] + k['k2b'] * 10**-(14 - pH)
                            k['k-2'] = k['k-2a'] * 10**-pH + k['k-2b']
                            k['k3'] = k['k3a'] + k['k3b'] * 10**-(14 - pH)
                            k['k-3'] = k['k-3a'] * 10**-pH + k['k-3b']
                            k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - pH)
                            k['k-4'] = k['k-4a'] * 10**-pH + k['k-4b']
                        
                            # Calculate θ values
                            theta = self.calculate_er_aom_theta(k)
                        
                            # Calculate r values
                            r = {
                                'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                'r2': k['k2'] * theta['theta*OH'] - k['k-2'] * theta['theta*O'],
                                'r3': k['k3'] * theta['theta*O'] - k['k-3'] * theta['theta*OOH'],
                                'r4': k['k4'] * theta['theta*OOH'] - k['k-4'] * theta['theta*']
                            }
                        
                            # Store results
                            k_values.append(k)
                            theta_values.append(theta)
                            r_values.append(r)
                    
                        # Prepare results for output
                        for i in range(1, 5):
                            results[f'k{i}'] = [k[f'k{i}'] for k in k_values]
                            results[f'k-{i}'] = [k[f'k-{i}'] for k in k_values]
                            results[f'lg(r{i})'] = [math.log10(abs(r[f'r{i}'])) if r[f'r{i}'] != 0 else -float('inf') for r in r_values]
                    
                        for theta_name in ['theta*', 'theta*OH', 'theta*O', 'theta*OOH']:
                            results[theta_name] = [theta[theta_name] for theta in theta_values]
            
                    else:  # Marcus-Gerischer kinetics
                        # Get step parameters
                        steps = []
                        for step_entry in self.er_aom_mg_entries:
                            step = {
                                'deltaG': float(step_entry['deltaG'].get()),
                                'lambda': float(step_entry['lambda'].get()),
                                'z': float(step_entry['z'].get())
                            }
                            steps.append(step)
                    
                        # Calculate for each variable value
                        k_values = []
                        theta_values = []
                        r_values = []
                    
                        for var in variable:
                            if self.variable_var.get() == "η":
                                eta = var
                                pH = fixed_value
                            else:
                                eta = fixed_value
                                pH = var
                        
                            # Calculate all k values
                            k = {}
                            for i, step in enumerate(steps, start=1):
                                # Marcus-Gerischer
                                # Forward reaction (a)
                                k[f'k{i}a'] = self.calculate_mg_ka(step, T, eta, pH)
                                # Backward reaction (-a)
                                k[f'k-{i}a'] = self.calculate_mg_k_minus_a(step, T, eta, pH)
                                # Forward reaction (b)
                                k[f'k{i}b'] = self.calculate_mg_kb(step, T, eta, pH)
                                # Backward reaction (-b)
                                k[f'k-{i}b'] = self.calculate_mg_k_minus_b(step, T, eta, pH)
                        
                            # Calculate combined k values
                            k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - pH)
                            k['k-1'] = k['k-1a'] * 10**-pH + k['k-1b']
                            k['k2'] = k['k2a'] + k['k2b'] * 10**-(14 - pH)
                            k['k-2'] = k['k-2a'] * 10**-pH + k['k-2b']
                            k['k3'] = k['k3a'] + k['k3b'] * 10**-(14 - pH)
                            k['k-3'] = k['k-3a'] * 10**-pH + k['k-3b']
                            k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - pH)
                            k['k-4'] = k['k-4a'] * 10**-pH + k['k-4b']
                        
                            # Calculate θ values
                            theta = self.calculate_er_aom_theta(k)
                        
                            # Calculate r values
                            r = {
                                'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                'r2': k['k2'] * theta['theta*OH'] - k['k-2'] * theta['theta*O'],
                                'r3': k['k3'] * theta['theta*O'] - k['k-3'] * theta['theta*OOH'],
                                'r4': k['k4'] * theta['theta*OOH'] - k['k-4'] * theta['theta*']
                            }
                        
                            # Store results
                            k_values.append(k)
                            theta_values.append(theta)
                            r_values.append(r)
                    
                        # Prepare results for output
                        for i in range(1, 5):
                            results[f'k{i}'] = [k[f'k{i}'] for k in k_values]
                            results[f'k-{i}'] = [k[f'k-{i}'] for k in k_values]
                            results[f'lg(r{i})'] = [math.log10(abs(r[f'r{i}'])) if r[f'r{i}'] != 0 else -float('inf') for r in r_values]
                    
                        for theta_name in ['theta*', 'theta*OH', 'theta*O', 'theta*OOH']:
                            results[theta_name] = [theta[theta_name] for theta in theta_values]
            
                else:  # LH-AOM model
                    if kinetics == "Butler-Volmer kinetics":
                        # Get parameters for LH-AOM with Butler-Volmer kinetics
                        ea0 = float(self.ea0_entry.get())
                    
                        # Get step parameters
                        steps = {}
                        for step_entry in self.lh_aom_bv_entries:
                            step_num = step_entry['step']
                            steps[step_num] = {
                                'deltaG': float(step_entry['deltaG'].get()),
                                'gamma': float(step_entry['gamma'].get()),
                                'beta': float(step_entry['beta'].get()),
                                'z': float(step_entry['z'].get())
                            }
                        required_steps = [1,21,22,31,32,4,5]
                        for step in required_steps:
                            if step not in steps:
                                raise ValueError(f"缺少步骤 {step} 的参数")
                    
                        # Calculate for each variable value
                        k_values = []
                        theta_values = []
                        r_values = []
                    
                        for var in variable:
                            if self.variable_var.get() == "η":
                                eta = var
                                pH = fixed_value
                            else:
                                eta = fixed_value
                                pH = var
                        
                            # Calculate all k values
                            k = {}
                        
                            # Step 1
                            step = steps[1]
                            k['k1a'] = self.calculate_bv_ka(step, ea0, T, eta, pH)
                            k['k-1a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, pH)
                            k['k1b'] = self.calculate_bv_kb(step, ea0, T, eta, pH)
                            k['k-1b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, pH)
                        
                            # Step 21
                            step = steps[21]
                            k['k21a'] = self.calculate_bv_ka(step, ea0, T, eta, pH)
                            k['k-21a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, pH)
                            k['k21b'] = self.calculate_bv_kb(step, ea0, T, eta, pH)
                            k['k-21b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, pH)
                        
                            # Step 22
                            step = steps[22]
                            k['k22a'] = self.calculate_bv_ka(step, ea0, T, eta, pH)
                            k['k-22a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, pH)
                            k['k22b'] = self.calculate_bv_kb(step, ea0, T, eta, pH)
                            k['k-22b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, pH)
                        
                            # Step 31
                            step = steps[31]
                            k['k31a'] = self.calculate_bv_ka(step, ea0, T, eta, pH)
                            k['k-31a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, pH)
                            k['k31b'] = self.calculate_bv_kb(step, ea0, T, eta, pH)
                            k['k-31b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, pH)
                        
                            # Step 32
                            step = steps[32]
                            k['k32a'] = self.calculate_bv_ka(step, ea0, T, eta, pH)
                            k['k-32a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, pH)
                            k['k32b'] = self.calculate_bv_kb(step, ea0, T, eta, pH)
                            k['k-32b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, pH)
                        
                            # Step 4
                            step = steps[4]
                            k['k4a'] = self.calculate_bv_ka(step, ea0, T, eta, pH)
                            k['k-4a'] = self.calculate_bv_k_minus_a(step, ea0, T, eta, pH)
                            k['k4b'] = self.calculate_bv_kb(step, ea0, T, eta, pH)
                            k['k-4b'] = self.calculate_bv_k_minus_b(step, ea0, T, eta, pH)
                        
                            # Step 5
                            step = steps[5]
                            gamma = step['gamma']
                            deltaG = step['deltaG']
                            if self.chem_method_var.get() == "BEP":
                                k['k5'] = (kB * T / h) * math.exp(-(ea0 + step['gamma'] * step['deltaG']) / (kB * T))
                                k['k-5'] = (kB * T / h) * math.exp(-(ea0 - step['gamma'] * step['deltaG']) / (kB * T))
                            else:
                                Ea5 = (1/gamma) * math.log(1 + (math.exp(gamma * deltaG)))
                                Ea_minus5 = (1/gamma) * math.log(1 + (math.exp(-gamma * deltaG)))
                                k['k5'] = (kB * T / h) * math.exp(-Ea5 / (kB * T))
                                k['k-5'] = (kB * T / h) * math.exp(-Ea_minus5 / (kB * T))

                            # Calculate combined k values
                            k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - pH)
                            k['k-1'] = k['k-1a'] * 10**-pH + k['k-1b']
                            k['k21'] = k['k21a'] + k['k21b'] * 10**-(14 - pH)
                            k['k-21'] = k['k-21a'] * 10**-pH + k['k-21b']
                            k['k22'] = k['k22a'] + k['k22b'] * 10**-(14 - pH)
                            k['k-22'] = k['k-22a'] * 10**-pH + k['k-22b']
                            k['k31'] = k['k31a'] + k['k31b'] * 10**-(14 - pH)
                            k['k-31'] = k['k-31a'] * 10**-pH + k['k-31b']
                            k['k32'] = k['k32a'] + k['k32b'] * 10**-(14 - pH)
                            k['k-32'] = k['k-32a'] * 10**-pH + k['k-32b']
                            k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - pH)
                            k['k-4'] = k['k-4a'] * 10**-pH + k['k-4b']
                            
                            # Calculate θ values
                            theta = self.calculate_lh_aom_theta(k)
                        
                            # Calculate r values
                            r = {
                                'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                'r21': k['k21'] * theta['theta*OH'] - k['k-21'] * theta['theta*(OH)2'],
                                'r22': k['k22'] * theta['theta*OH'] - k['k-22'] * theta['theta*O'],
                                'r31': k['k31'] * theta['theta*(OH)2'] - k['k-31'] * theta['theta*O(OH)'],
                                'r32': k['k32'] * theta['theta*O'] - k['k-32'] * theta['theta*O(OH)'],
                                'r4': k['k4'] * theta['theta*O(OH)'] - k['k-4'] * theta['theta*O(O)'],
                                'r5': k['k5'] * theta['theta*O(O)'] - k['k-5'] * theta['theta*']
                            }
                            # 在计算r值的代码块末尾添加：
                            r['r2'] = r['r21'] + r['r22']
                            r['r3'] = r['r31'] + r['r32']
                            
                            # Store results
                            k_values.append(k)
                            theta_values.append(theta)
                            r_values.append(r)
                    
                        # Prepare results for output
                        for step in ['1', '21', '22', '31', '32', '4', '5']:
                            results[f'k{step}'] = [k[f'k{step}'] for k in k_values]
                            results[f'k-{step}'] = [k[f'k-{step}'] for k in k_values]
                    
                        for r_step in ['1', '21', '22', '2', '31', '32','3',  '4', '5']:
                            results[f'r{r_step}'] = [r[f'r{r_step}'] for r in r_values]  # 添加原始速率
                            results['lg(r5)'] = [math.log10(abs(r['r5'])) if r['r5'] != 0 else -float('inf') for r in r_values]
                            results['lg(r21)'] = [math.log10(abs(r['r21'])) if r['r21'] != 0 else -float('inf') for r in r_values]
                            results['lg(r22)'] = [math.log10(abs(r['r22'])) if r['r22'] != 0 else -float('inf') for r in r_values]
                     
                        for theta_name in ['theta*', 'theta*OH', 'theta*(OH)2', 'theta*O', 'theta*O(OH)', 'theta*O(O)']:
                            results[theta_name] = [theta[theta_name] for theta in theta_values]
                
                    elif kinetics == "Marcus kinetics":
                        # 获取 Ea,0
                        # Get parameters for LH-AOM with Butler-Volmer kinetics
                        ea0 = float(self.ea0_entry.get())
                    
                        # Get step parameters
                        steps = {}
                        for step_entry in self.lh_aom_marcus_entries:
                            step_num = step_entry['step']
                            steps[step_num] = {
                                'deltaG': float(step_entry['deltaG'].get()),
                                'lambda': float(step_entry['lambda'].get()),
                                'gamma': float(step_entry['gamma'].get()),
                                'z': float(step_entry['z'].get())
                            }
                        required_steps = [1,21,22,31,32,4,5]
                        for step in required_steps:
                            if step not in steps:
                                raise ValueError(f"缺少步骤 {step} 的参数")
                    
                        # Calculate for each variable value
                        k_values = []
                        theta_values = []
                        r_values = []
                    
                        for var in variable:
                            if self.variable_var.get() == "η":
                                eta = var
                                pH = fixed_value
                            else:
                                eta = fixed_value
                                pH = var
                        
                            # Calculate all k values
                            k = {}
                        
                            #"Marcus kinetics"
                            # Step 1
                            step = steps[1]                            
                            k['k1a'] = self.calculate_marcus_ka(step, T, eta, pH)
                            k['k-1a'] = self.calculate_marcus_k_minus_a(step, T, eta, pH)
                            k['k1b'] = self.calculate_marcus_kb(step, T, eta, pH)
                            k['k-1b'] = self.calculate_marcus_k_minus_b(step, T, eta, pH)
                            
                            # Step 21
                            step = steps[21]
                            k['k21a'] = self.calculate_marcus_ka(step, T, eta, pH)
                            k['k-21a'] = self.calculate_marcus_k_minus_a(step, T, eta, pH)
                            k['k21b'] = self.calculate_marcus_kb(step, T, eta, pH)
                            k['k-21b'] = self.calculate_marcus_k_minus_b(step, T, eta, pH)
                          
                            # Step 22
                            step = steps[22]
                            k['k22a'] = self.calculate_marcus_ka(step, T, eta, pH)
                            k['k-22a'] = self.calculate_marcus_k_minus_a(step, T, eta, pH)
                            k['k22b'] = self.calculate_marcus_kb(step, T, eta, pH)
                            k['k-22b'] = self.calculate_marcus_k_minus_b(step, T, eta, pH)
                            
                            # Step 31
                            step = steps[31]
                            k['k31a'] = self.calculate_marcus_ka(step, T, eta, pH)
                            k['k-31a'] = self.calculate_marcus_k_minus_a(step, T, eta, pH)
                            k['k31b'] = self.calculate_marcus_kb(step, T, eta, pH)
                            k['k-31b'] = self.calculate_marcus_k_minus_b(step, T, eta, pH)
                     
                            # Step 32
                            step = steps[32]                  
                            k['k32a'] = self.calculate_marcus_ka(step, T, eta, pH)
                            k['k-32a'] = self.calculate_marcus_k_minus_a(step, T, eta, pH)
                            k['k32b'] = self.calculate_marcus_kb(step, T, eta, pH)
                            k['k-32b'] = self.calculate_marcus_k_minus_b(step, T, eta, pH)
                           
                            # Step 4
                            step = steps[4]                         
                            k['k4a'] = self.calculate_marcus_ka(step, T, eta, pH)
                            k['k-4a'] = self.calculate_marcus_k_minus_a(step, T, eta, pH)
                            k['k4b'] = self.calculate_marcus_kb(step, T, eta, pH)
                            k['k-4b'] = self.calculate_marcus_k_minus_b(step, T, eta, pH)
                            
                            # Step 5
                            step = steps[5]
                            gamma = step['gamma']
                            deltaG = step['deltaG']
                            if self.chem_method_var.get() == "BEP":
                                k['k5'] = (kB * T / h) * math.exp(-(ea0 + step['gamma'] * step['deltaG']) / (kB * T))
                                k['k-5'] = (kB * T / h) * math.exp(-(ea0 - step['gamma'] * step['deltaG']) / (kB * T))
                            else:
                                Ea5 = (1/gamma) * math.log(1 + (math.exp(gamma * deltaG)))
                                Ea_minus5 = (1/gamma) * math.log(1 + (math.exp(-gamma * deltaG)))
                                k['k5'] = (kB * T / h) * math.exp(-Ea5 / (kB * T))
                                k['k-5'] = (kB * T / h) * math.exp(-Ea_minus5 / (kB * T))
                 
                            # Calculate combined k values
                            k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - pH)
                            k['k-1'] = k['k-1a'] * 10**-pH + k['k-1b']
                            k['k21'] = k['k21a'] + k['k21b'] * 10**-(14 - pH)
                            k['k-21'] = k['k-21a'] * 10**-pH + k['k-21b']
                            k['k22'] = k['k22a'] + k['k22b'] * 10**-(14 - pH)
                            k['k-22'] = k['k-22a'] * 10**-pH + k['k-22b']
                            k['k31'] = k['k31a'] + k['k31b'] * 10**-(14 - pH)
                            k['k-31'] = k['k-31a'] * 10**-pH + k['k-31b']
                            k['k32'] = k['k32a'] + k['k32b'] * 10**-(14 - pH)
                            k['k-32'] = k['k-32a'] * 10**-pH + k['k-32b']
                            k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - pH)
                            k['k-4'] = k['k-4a'] * 10**-pH + k['k-4b']
                        
                            # Calculate θ values
                            theta = self.calculate_lh_aom_theta(k)
                        
                           # Calculate r values
                            r = {
                                'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                'r21': k['k21'] * theta['theta*OH'] - k['k-21'] * theta['theta*(OH)2'],
                                'r22': k['k22'] * theta['theta*OH'] - k['k-22'] * theta['theta*O'],
                                'r31': k['k31'] * theta['theta*(OH)2'] - k['k-31'] * theta['theta*O(OH)'],
                                'r32': k['k32'] * theta['theta*O'] - k['k-32'] * theta['theta*O(OH)'],
                                'r4': k['k4'] * theta['theta*O(OH)'] - k['k-4'] * theta['theta*O(O)'],
                                'r5': k['k5'] * theta['theta*O(O)'] - k['k-5'] * theta['theta*']
                            }
                            # 在计算r值的代码块末尾添加：
                            r['r2'] = r['r21'] + r['r22']
                            r['r3'] = r['r31'] + r['r32']
                        
                            # Store results
                            k_values.append(k)
                            theta_values.append(theta)
                            r_values.append(r)
                    
                        # Prepare results for output
                        for step in ['1', '21', '22', '31', '32', '4', '5']:
                            results[f'k{step}'] = [k[f'k{step}'] for k in k_values]
                            results[f'k-{step}'] = [k[f'k-{step}'] for k in k_values]
                    
                        for r_step in ['1', '21', '22', '2', '31', '32','3',  '4', '5']:
                            results[f'r{r_step}'] = [r[f'r{r_step}'] for r in r_values]  # 添加原始速率
                            results['lg(r5)'] = [math.log10(abs(r['r5'])) if r['r5'] != 0 else -float('inf') for r in r_values]
                            results['lg(r21)'] = [math.log10(abs(r['r21'])) if r['r21'] != 0 else -float('inf') for r in r_values]
                            results['lg(r22)'] = [math.log10(abs(r['r22'])) if r['r22'] != 0 else -float('inf') for r in r_values]
                     
                        for theta_name in ['theta*', 'theta*OH', 'theta*(OH)2', 'theta*O', 'theta*O(OH)', 'theta*O(O)']:
                            results[theta_name] = [theta[theta_name] for theta in theta_values]

                    else:  # Marcus-Gerischer kinetics
                        # 获取 Ea,0
                        ea0 = float(self.ea0_entry.get())

                        # Get step parameters
                        steps = {}
                        for step_entry in self.lh_aom_mg_entries:
                            step_num = step_entry['step']
                            steps[step_num] = {
                                'deltaG': float(step_entry['deltaG'].get()),
                                'lambda': float(step_entry['lambda'].get()),
                                'gamma': float(step_entry['gamma'].get()),
                                'z': float(step_entry['z'].get())
                            }
                        required_steps = [1,21,22,31,32,4,5]
                        for step in required_steps:
                            if step not in steps:
                                raise ValueError(f"缺少步骤 {step} 的参数")
                    
                        # Calculate for each variable value
                        k_values = []
                        theta_values = []
                        r_values = []
                    
                        for var in variable:
                            if self.variable_var.get() == "η":
                                eta = var
                                pH = fixed_value
                            else:
                                eta = fixed_value
                                pH = var
                        
                            # Calculate all k values
                            k = {}
                        
                            # Step 1
                            step = steps[1]                           
                            k['k1a'] = self.calculate_mg_ka(step, T, eta, pH)
                            k['k-1a'] = self.calculate_mg_k_minus_a(step, T, eta, pH)
                            k['k1b'] = self.calculate_mg_kb(step, T, eta, pH)
                            k['k-1b'] = self.calculate_mg_k_minus_b(step, T, eta, pH)
                        
                            # Step 21
                            step = steps[21]
                            k['k21a'] = self.calculate_mg_ka(step, T, eta, pH)
                            k['k-21a'] = self.calculate_mg_k_minus_a(step, T, eta, pH)
                            k['k21b'] = self.calculate_mg_kb(step, T, eta, pH)
                            k['k-21b'] = self.calculate_mg_k_minus_b(step, T, eta, pH)
                        
                            # Step 22
                            step = steps[22]
                            k['k22a'] = self.calculate_mg_ka(step, T, eta, pH)
                            k['k-22a'] = self.calculate_mg_k_minus_a(step, T, eta, pH)
                            k['k22b'] = self.calculate_mg_kb(step, T, eta, pH)
                            k['k-22b'] = self.calculate_mg_k_minus_b(step, T, eta, pH)
                        
                            # Step 31
                            step = steps[31]
                            k['k31a'] = self.calculate_mg_ka(step, T, eta, pH)
                            k['k-31a'] = self.calculate_mg_k_minus_a(step, T, eta, pH)
                            k['k31b'] = self.calculate_mg_kb(step, T, eta, pH)
                            k['k-31b'] = self.calculate_mg_k_minus_b(step, T, eta, pH)
                        
                            # Step 32
                            step = steps[32]
                            k['k32a'] = self.calculate_mg_ka(step, T, eta, pH)
                            k['k-32a'] = self.calculate_mg_k_minus_a(step, T, eta, pH)
                            k['k32b'] = self.calculate_mg_kb(step, T, eta, pH)
                            k['k-32b'] = self.calculate_mg_k_minus_b(step, T, eta, pH)
                        
                            # Step 4
                            step = steps[4]
                            k['k4a'] = self.calculate_mg_ka(step, T, eta, pH)
                            k['k-4a'] = self.calculate_mg_k_minus_a(step, T, eta, pH)
                            k['k4b'] = self.calculate_mg_kb(step, T, eta, pH)
                            k['k-4b'] = self.calculate_mg_k_minus_b(step, T, eta, pH)
                        
                            # Step 5
                            step = steps[5]
                            gamma = step['gamma']
                            deltaG = step['deltaG']
                            if self.chem_method_var.get() == "BEP":
                                k['k5'] = (kB * T / h) * math.exp(-(ea0 + step['gamma'] * step['deltaG']) / (kB * T))
                                k['k-5'] = (kB * T / h) * math.exp(-(ea0 - step['gamma'] * step['deltaG']) / (kB * T))
                            else:
                                Ea5 = (1/gamma) * math.log(1 + (math.exp(gamma * deltaG)))
                                Ea_minus5 = (1/gamma) * math.log(1 + (math.exp(-gamma * deltaG)))
                                k['k5'] = (kB * T / h) * math.exp(-Ea5 / (kB * T))
                                k['k-5'] = (kB * T / h) * math.exp(-Ea_minus5 / (kB * T))
                
                            # Calculate combined k values
                            k['k1'] = k['k1a'] + k['k1b'] * 10**-(14 - pH)
                            k['k-1'] = k['k-1a'] * 10**-pH + k['k-1b']
                            k['k21'] = k['k21a'] + k['k21b'] * 10**-(14 - pH)
                            k['k-21'] = k['k-21a'] * 10**-pH + k['k-21b']
                            k['k22'] = k['k22a'] + k['k22b'] * 10**-(14 - pH)
                            k['k-22'] = k['k-22a'] * 10**-pH + k['k-22b']
                            k['k31'] = k['k31a'] + k['k31b'] * 10**-(14 - pH)
                            k['k-31'] = k['k-31a'] * 10**-pH + k['k-31b']
                            k['k32'] = k['k32a'] + k['k32b'] * 10**-(14 - pH)
                            k['k-32'] = k['k-32a'] * 10**-pH + k['k-32b']
                            k['k4'] = k['k4a'] + k['k4b'] * 10**-(14 - pH)
                            k['k-4'] = k['k-4a'] * 10**-pH + k['k-4b']
                        
                            # Calculate θ values
                            theta = self.calculate_lh_aom_theta(k)
                        
                           # Calculate r values
                            r = {
                                'r1': k['k1'] * theta['theta*'] - k['k-1'] * theta['theta*OH'],
                                'r21': k['k21'] * theta['theta*OH'] - k['k-21'] * theta['theta*(OH)2'],
                                'r22': k['k22'] * theta['theta*OH'] - k['k-22'] * theta['theta*O'],
                                'r31': k['k31'] * theta['theta*(OH)2'] - k['k-31'] * theta['theta*O(OH)'],
                                'r32': k['k32'] * theta['theta*O'] - k['k-32'] * theta['theta*O(OH)'],
                                'r4': k['k4'] * theta['theta*O(OH)'] - k['k-4'] * theta['theta*O(O)'],
                                'r5': k['k5'] * theta['theta*O(O)'] - k['k-5'] * theta['theta*']
                            }
                            # 在计算r值的代码块末尾添加：
                            r['r2'] = r['r21'] + r['r22']
                            r['r3'] = r['r31'] + r['r32']
                        
                            # Store results
                            k_values.append(k)
                            theta_values.append(theta)
                            r_values.append(r)
                    
                        # Prepare results for output
                        for step in ['1', '21', '22', '31', '32', '4', '5']:
                            results[f'k{step}'] = [k[f'k{step}'] for k in k_values]
                            results[f'k-{step}'] = [k[f'k-{step}'] for k in k_values]
                    
                        for r_step in ['1', '21', '22', '2', '31', '32','3',  '4', '5']:
                            results[f'r{r_step}'] = [r[f'r{r_step}'] for r in r_values]  # 添加原始速率
                            results['lg(r5)'] = [math.log10(abs(r['r5'])) if r['r5'] != 0 else -float('inf') for r in r_values]
                            results['lg(r21)'] = [math.log10(abs(r['r21'])) if r['r21'] != 0 else -float('inf') for r in r_values]
                            results['lg(r22)'] = [math.log10(abs(r['r22'])) if r['r22'] != 0 else -float('inf') for r in r_values]
                     
                        for theta_name in ['theta*', 'theta*OH', 'theta*(OH)2', 'theta*O', 'theta*O(OH)', 'theta*O(O)']:
                            results[theta_name] = [theta[theta_name] for theta in theta_values]
                            
                # 获取当前参数
                model = self.model_var.get()
                kinetics = self.kinetics_var.get()
                T = float(self.temp_entry.get())
                parameters = {
                    "Model": model,
                    "Kinetics": kinetics,
                    "Temperature (K)": T,
                }

                # 将参数添加到结果DataFrame中
                for key, value in parameters.items():
                    results[key] = [value] * len(var_range)

                # 创建结果DataFrame
                self.results_df = pd.DataFrame(results)
                
                # 更新主窗口的表格和图表
                self.update_results_table()
                self.update_plot()

                # 在新窗口中显示图表
                self.create_plot_window()
                self.update_plot_in_new_window()

            messagebox.showinfo("计算完成", "计算成功完成！")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"An error occurred during calculation:\n{str(e)}")
            traceback.print_exc()
            
    def create_contour_plot(self):
        """创建二维等值线图"""
        if not hasattr(self, 'results_2d'):
            return
        
        # 创建新窗口
        contour_window = tk.Toplevel(self.root)
        contour_window.title("二维等值线图")
        contour_window.geometry("800x600")
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # 提取数据
        eta = self.results_2d['eta']
        ph = self.results_2d['ph']
        lgr = self.results_2d['lgr']
        
        # 绘制等值线图
        contour = ax.contourf(eta, ph, lgr, levels=20, cmap='viridis')
        fig.colorbar(contour, ax=ax, label='log(r5)')
        
        ax.set_xlabel('η (V)')
        ax.set_ylabel('pH')
        ax.set_title('log(r5) as function of η and pH')
        
        canvas = FigureCanvasTkAgg(fig, master=contour_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, contour_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

   # 辅助计算函数（完整实现）
    def calculate_bv_ka(self, step, ea0, T, eta, pH):
        """Calculate k for forward reaction (a) using Butler-Volmer kinetics"""
        if self.bv_method_var.get() == "BEP":
            # 原有BEP公式
            ea0 = float(self.ea0_entry.get())
            term1 = (kB * T / h) * math.exp(-(ea0 + step['gamma'] * step['deltaG']) / (kB * T))
            term2 = math.exp(step['beta'] * (F / (R * T)) * (eta - (R * T / F) * (math.log(10)) * pH))
            return term1 * term2
        else:
            # Softplus公式
            gamma = step['gamma']
            deltaG = step['deltaG']
            # 计算Ea0
            Ea0 = (1/gamma) * math.log(1 + math.exp(gamma * deltaG))
            # 计算k_ia
            term1 = (kB * T / h) * math.exp(-Ea0 / (kB * T))
            term2 = math.exp(step['beta'] * (F / (R * T)) * (eta - (R * T / F) * (math.log(10)) * pH))
            return term1 * term2

    def calculate_bv_k_minus_a(self, step, ea0, T, eta, pH):
        """Calculate k for backward reaction (-a) using Butler-Volmer kinetics"""
        if self.bv_method_var.get() == "BEP":
            # 原有BEP公式
            ea0 = float(self.ea0_entry.get())
            term1 = (kB * T / h) * math.exp(-(ea0 - step['gamma'] * step['deltaG']) / (kB * T))
            term2 = math.exp(- ( 1 - step['beta']) * (F / (R * T)) * (eta - (R * T / F) * (math.log(10)) * pH))
            return term1 * term2
        else:
            # Softplus公式
            gamma = step['gamma']
            deltaG = step['deltaG']
            # 计算Ea0
            Ea0 = (1/gamma) * math.log(1 + math.exp(gamma * deltaG))
            # 计算k_minus_ia
            term1 = (kB * T / h) * math.exp(-Ea0 / (kB * T))
            term2 = math.exp(- ( 1 - step['beta']) * (F / (R * T)) * (eta - (R * T / F) * (math.log(10)) * pH))
            return term1 * term2

    def calculate_bv_kb(self, step, ea0, T, eta, pH):
        """Calculate k for forward reaction (b) using Butler-Volmer kinetics"""
        deltaGw = self.delta_gw_var.get()  # 获取用户输入
        if self.bv_method_var.get() == "BEP":
            # 原有BEP公式
            ea0 = float(self.ea0_entry.get())
            term1 = (kB * T / h) * math.exp((-(ea0 + step['gamma'] * (step['deltaG'] - step['z'] * deltaGw ))) / (kB * T))
            term2 = math.exp(step['beta'] * (F / (R * T)) * (eta - (R * T / F) * (math.log(10)) * pH))
            return term1 * term2
        else:
            # Softplus公式
            gamma = step['gamma']
            deltaG = step['deltaG']
            # 计算Ea0
            Ea0 = (1/gamma) * math.log(1 + math.exp(gamma * (deltaG + step['z'] * (R * T / F) * math.log(10) * (-14))))
            # 计算k_ib
            term1 = (kB * T / h) * math.exp(-Ea0 / (kB * T))
            term2 = math.exp(step['beta'] * (F / (R * T)) * (eta - (R * T / F) * (math.log(10)) * pH))
            return term1 * term2

    def calculate_bv_k_minus_b(self, step, ea0, T, eta, pH):
        """Calculate k for backward reaction (-b) using Butler-Volmer kinetics"""
        deltaGw = self.delta_gw_var.get()  # 获取用户输入
        if self.bv_method_var.get() == "BEP":
            # 原有BEP公式
            ea0 = float(self.ea0_entry.get())
            term1 = (kB * T / h) * math.exp((-(ea0 - step['gamma'] * (step['deltaG'] - step['z'] * deltaGw))) / (kB * T))
            term2 = math.exp(- ( 1 - step['beta']) * (F / (R * T)) * (eta - (R * T / F) * (math.log(10)) * pH))
            return term1 * term2
        else:
            # Softplus公式
            gamma = step['gamma']
            deltaG = step['deltaG']
            # 计算Ea0
            Ea0 = (1/gamma) * math.log(1 + math.exp(gamma * ( - deltaG - step['z'] * (R * T / F) * (math.log(10)) * (-14))))
            # 计算k_minus_ib
            term1 = (kB * T / h) * math.exp(-Ea0 / (kB * T))
            term2 = math.exp(- ( 1 - step['beta']) * (F / (R * T)) * (eta - (R * T / F) * (math.log(10)) * pH))
            return term1 * term2

    def calculate_marcus_ka(self, step, T, eta, pH):
        """Calculate k for forward reaction (a) using Marcus kinetics"""
        exponent = -(+step['deltaG'] - step['z'] * (eta - (R * T / F) * (math.log(10)) * pH) + step['lambda'])**2 / (4 * step['lambda'] * kB * T)
        return (kB * T / h) * math.exp(exponent)

    def calculate_marcus_k_minus_a(self, step, T, eta, pH):
        """Calculate k for backward reaction (-a) using Marcus kinetics"""
        exponent = -(-step['deltaG'] + step['z'] * (eta - (R * T / F) * (math.log(10)) * pH) + step['lambda'])**2 / (4 * step['lambda'] * kB * T)
        return (kB * T / h) * math.exp(exponent)

    def calculate_marcus_kb(self, step, T, eta, pH):
        """Calculate k for forward reaction (b) using Marcus kinetics"""
        deltaGw = self.delta_gw_var.get()  # 获取用户输入
        exponent = -(+step['deltaG'] - step['z'] * (eta - (R * T / F) * (math.log(10)) * pH + deltaGw ) + step['lambda'])**2 / (4 * step['lambda'] * kB * T)
        return (kB * T / h) * math.exp(exponent)

    def calculate_marcus_k_minus_b(self, step, T, eta, pH):
        """Calculate k for backward reaction (-b) using Marcus kinetics"""
        deltaGw = self.delta_gw_var.get()  # 获取用户输入
        exponent = -(-step['deltaG'] + step['z'] * (eta - (R * T / F) * (math.log(10)) * pH + deltaGw) + step['lambda'])**2 / (4 * step['lambda'] * kB * T)
        return (kB * T / h) * math.exp(exponent)

    def calculate_mg_ka(self, step, T, eta, pH):
        """Calculate k for forward reaction (a) using Marcus-Gerischer kinetics"""    
        deltaG = step['deltaG']    
        z = step['z']
        lambda_val = step['lambda']
    
        def integrand(epsilon):
            term = deltaG - z * (eta - (R * T / F) * (math.log(10)) * pH) 
            exponent = -( (term + epsilon + lambda_val) ** 2 ) / (4 * lambda_val * kB * T)
            fermi = 1 / (1 + math.exp(-epsilon / (kB * T ))) 
            return math.exp(exponent) * fermi
        
        int_low = -5 * lambda_val
        int_high = 5 * lambda_val
        integral, _ = quad(integrand, int_low, int_high)
        return (kB * T / h) * integral

    def calculate_mg_k_minus_a(self, step, T, eta, pH):
        """Calculate k for backward reaction (-a) using Marcus-Gerischer kinetics"""
        deltaG = step['deltaG']
        z = step['z']
        lambda_val = step['lambda']
    
        def integrand(epsilon):
            term = -deltaG + z * (eta - (R * T / F) * (math.log(10)) * pH)
            exponent = -( (term - epsilon + lambda_val) ** 2 ) / (4 * lambda_val * kB * T)
            fermi = 1 / (1 + math.exp(epsilon / (kB * T )))
            return math.exp(exponent) * fermi

        int_low = -5 * lambda_val
        int_high = 5 * lambda_val
        integral, _ = quad(integrand, int_low, int_high)
        return (kB * T / h) * integral

    def calculate_mg_kb(self, step, T, eta, pH):
        """Calculate k for forward reaction (b) using Marcus-Gerischer kinetics"""
        deltaGw = self.delta_gw_var.get()  # 获取用户输入
        deltaG = step['deltaG']
        z = step['z']
        lambda_val = step['lambda']
    
        def integrand(epsilon):
            term = deltaG - z * (eta - (R * T / F) * (math.log(10)) * pH + deltaGw)
            exponent = -( (term + epsilon + lambda_val) ** 2 ) / (4 * lambda_val * kB * T)
            fermi = 1 / (1 + math.exp(-epsilon / (kB * T )))
            return math.exp(exponent) * fermi
    
        int_low = -5 * lambda_val
        int_high = 5 * lambda_val
        integral, _ = quad(integrand, int_low, int_high)
        return (kB * T / h) * integral

    def calculate_mg_k_minus_b(self, step, T, eta, pH):
        """Calculate k for backward reaction (-b) using Marcus-Gerischer kinetics"""
        deltaGw = self.delta_gw_var.get()  # 获取用户输入
        deltaG = step['deltaG']
        z = step['z']
        lambda_val = step['lambda']
    
        def integrand(epsilon):
            term = - deltaG + z * (eta - (R * T / F) * (math.log(10)) * pH + deltaGw)
            exponent = -( (term - epsilon + lambda_val) ** 2 ) / (4 * lambda_val * kB * T)
            fermi = 1 / (1 + math.exp(epsilon / (kB * T )))
            return math.exp(exponent) * fermi
    
        int_low = -5 * lambda_val
        int_high = 5 * lambda_val
        integral, _ = quad(integrand, int_low, int_high)
        return (kB * T / h) * integral


    def calculate_er_aom_theta(self, k):
        """Calculate θ values for ER-AOM model"""
        # Calculate numerator and denominator terms
        term1 = k['k-1']*k['k-2']*k['k-3'] + k['k-1']*k['k-2']*k['k4'] + k['k-1']*k['k3']*k['k4'] + k['k2']*k['k3']*k['k4']
        term2 = k['k1']*k['k-2']*k['k-3'] + k['k1']*k['k-2']*k['k4'] + k['k1']*k['k3']*k['k4'] + k['k-2']*k['k-3']*k['k-4']
        term3 = k['k1']*k['k2']*k['k-3'] + k['k1']*k['k2']*k['k4'] + k['k-1']*k['k-3']*k['k-4'] + k['k2']*k['k-3']*k['k-4']
        term4 = k['k1']*k['k2']*k['k3'] + k['k-1']*k['k-2']*k['k-4'] + k['k-1']*k['k3']*k['k-4'] + k['k2']*k['k3']*k['k-4']
        
        denominator = term1 + term2 + term3 + term4
        if denominator <= 1e-30:  # 避免浮点精度问题
           raise ValueError("分母为零，无法计算θ。请检查步骤参数是否导致速率常数（k）全为零。")
        
        # Calculate θ values
        theta = {
            'theta*': term1 / denominator,
            'theta*OH': term2 / denominator,
            'theta*O': term3 / denominator,
            'theta*OOH': term4 / denominator
        }
        
        return theta

    def calculate_lh_aom_theta(self, k):
        """Calculate θ values for LH-AOM model"""
        # Calculate numerator and denominator terms
        term1 = k['k-1']*k['k-21']*k['k-22']*k['k-31']*k['k-4'] + k['k-1']*k['k-21']*k['k-22']*k['k-31']*k['k5'] + k['k-1']*k['k-21']*k['k-22']*k['k-32']*k['k-4'] + k['k-1']*k['k-21']*k['k-22']*k['k-32']*k['k5'] + k['k-1']*k['k-21']*k['k-22']*k['k4']*k['k5'] + k['k-1']*k['k-21']*k['k-31']*k['k32']*k['k-4'] + k['k-1']*k['k-21']*k['k-31']*k['k32']*k['k5'] + k['k-1']*k['k-21']*k['k32']*k['k4']*k['k5'] + k['k-1']*k['k-22']*k['k31']*k['k-32']*k['k-4'] + k['k-1']*k['k-22']*k['k31']*k['k-32']*k['k5'] + k['k-1']*k['k-22']*k['k31']*k['k4']*k['k5'] + k['k-1']*k['k31']*k['k32']*k['k4']*k['k5'] + k['k21']*k['k-22']*k['k31']*k['k4']*k['k5'] + k['k21']*k['k31']*k['k32']*k['k4']*k['k5'] + k['k-21']*k['k22']*k['k32']*k['k4']*k['k5'] + k['k22']*k['k31']*k['k32']*k['k4']*k['k5']
        term2 = k['k1']*k['k-21']*k['k-22']*k['k-31']*k['k-4'] + k['k1']*k['k-21']*k['k-22']*k['k-31']*k['k5'] + k['k1']*k['k-21']*k['k-22']*k['k-32']*k['k-4'] + k['k1']*k['k-21']*k['k-22']*k['k-32']*k['k5'] + k['k1']*k['k-21']*k['k-22']*k['k4']*k['k5'] + k['k1']*k['k-21']*k['k-31']*k['k32']*k['k-4'] + k['k1']*k['k-21']*k['k-31']*k['k32']*k['k5'] + k['k1']*k['k-21']*k['k32']*k['k4']*k['k5'] + k['k1']*k['k-22']*k['k31']*k['k-32']*k['k-4'] + k['k1']*k['k-22']*k['k31']*k['k-32']*k['k5'] + k['k1']*k['k-22']*k['k31']*k['k4']*k['k5'] + k['k1']*k['k31']*k['k32']*k['k4']*k['k5'] + k['k-21']*k['k-22']*k['k-31']*k['k-4']*k['k-5'] + k['k-21']*k['k-22']*k['k-32']*k['k-4']*k['k-5'] + k['k-21']*k['k-31']*k['k32']*k['k-4']*k['k-5'] + k['k-22']*k['k31']*k['k-32']*k['k-4']*k['k-5']
        term3 = k['k1']*k['k21']*k['k-22']*k['k-31']*k['k-4'] + k['k1']*k['k21']*k['k-22']*k['k-31']*k['k5'] + k['k1']*k['k21']*k['k-22']*k['k-32']*k['k-4'] + k['k1']*k['k21']*k['k-22']*k['k-32']*k['k5'] + k['k1']*k['k21']*k['k-22']*k['k4']*k['k5'] + k['k1']*k['k21']*k['k-31']*k['k32']*k['k-4'] + k['k1']*k['k21']*k['k-31']*k['k32']*k['k5'] + k['k1']*k['k21']*k['k32']*k['k4']*k['k5'] + k['k1']*k['k22']*k['k-31']*k['k32']*k['k-4'] + k['k1']*k['k22']*k['k-31']*k['k32']*k['k5'] + k['k-1']*k['k-22']*k['k-31']*k['k-4']*k['k-5'] + k['k-1']*k['k-31']*k['k32']*k['k-4']*k['k-5'] + k['k21']*k['k-22']*k['k-31']*k['k-4']*k['k-5'] + k['k21']*k['k-22']*k['k-32']*k['k-4']*k['k-5'] + k['k21']*k['k-31']*k['k32']*k['k-4']*k['k-5'] + k['k22']*k['k-31']*k['k32']*k['k-4']*k['k-5']
        term4 = k['k1']*k['k21']*k['k31']*k['k-32']*k['k-4'] + k['k1']*k['k21']*k['k31']*k['k-32']*k['k5'] + k['k1']*k['k-21']*k['k22']*k['k-31']*k['k-4'] + k['k1']*k['k-21']*k['k22']*k['k-31']*k['k5'] + k['k1']*k['k-21']*k['k22']*k['k-32']*k['k-4'] + k['k1']*k['k-21']*k['k22']*k['k-32']*k['k5'] + k['k1']*k['k-21']*k['k22']*k['k4']*k['k5'] + k['k1']*k['k22']*k['k31']*k['k-32']*k['k-4'] + k['k1']*k['k22']*k['k31']*k['k-32']*k['k5'] + k['k1']*k['k22']*k['k31']*k['k4']*k['k5'] + k['k-1']*k['k-21']*k['k-32']*k['k-4']*k['k-5'] + k['k-1']*k['k31']*k['k-32']*k['k-4']*k['k-5'] + k['k21']*k['k31']*k['k-32']*k['k-4']*k['k-5'] + k['k-21']*k['k22']*k['k-31']*k['k-4']*k['k-5'] + k['k-21']*k['k22']*k['k-32']*k['k-4']*k['k-5'] + k['k22']*k['k31']*k['k-32']*k['k-4']*k['k-5']
        term5 = k['k1']*k['k21']*k['k-22']*k['k31']*k['k-4'] + k['k1']*k['k21']*k['k-22']*k['k31']*k['k5'] + k['k1']*k['k21']*k['k31']*k['k32']*k['k-4'] + k['k1']*k['k21']*k['k31']*k['k32']*k['k5'] + k['k1']*k['k-21']*k['k22']*k['k32']*k['k-4'] + k['k1']*k['k-21']*k['k22']*k['k32']*k['k5'] + k['k1']*k['k22']*k['k31']*k['k32']*k['k-4'] + k['k1']*k['k22']*k['k31']*k['k32']*k['k5'] + k['k-1']*k['k-21']*k['k-22']*k['k-4']*k['k-5'] + k['k-1']*k['k-21']*k['k32']*k['k-4']*k['k-5'] + k['k-1']*k['k-22']*k['k31']*k['k-4']*k['k-5'] + k['k-1']*k['k31']*k['k32']*k['k-4']*k['k-5'] + k['k21']*k['k-22']*k['k31']*k['k-4']*k['k-5'] + k['k21']*k['k31']*k['k32']*k['k-4']*k['k-5'] + k['k-21']*k['k22']*k['k32']*k['k-4']*k['k-5'] + k['k22']*k['k31']*k['k32']*k['k-4']*k['k-5']
        term6 = k['k1']*k['k21']*k['k-22']*k['k31']*k['k4'] + k['k1']*k['k21']*k['k31']*k['k32']*k['k4'] + k['k1']*k['k-21']*k['k22']*k['k32']*k['k4'] + k['k1']*k['k22']*k['k31']*k['k32']*k['k4'] + k['k-1']*k['k-21']*k['k-22']*k['k-31']*k['k-5'] + k['k-1']*k['k-21']*k['k-22']*k['k-32']*k['k-5'] + k['k-1']*k['k-21']*k['k-22']*k['k4']*k['k-5'] + k['k-1']*k['k-21']*k['k-31']*k['k32']*k['k-5'] + k['k-1']*k['k-21']*k['k32']*k['k4']*k['k-5'] + k['k-1']*k['k-22']*k['k31']*k['k-32']*k['k-5'] + k['k-1']*k['k-22']*k['k31']*k['k4']*k['k-5'] + k['k-1']*k['k31']*k['k32']*k['k4']*k['k-5'] + k['k21']*k['k-22']*k['k31']*k['k4']*k['k-5'] + k['k21']*k['k31']*k['k32']*k['k4']*k['k-5'] + k['k-21']*k['k22']*k['k32']*k['k4']*k['k-5'] + k['k22']*k['k31']*k['k32']*k['k4']*k['k-5']
     
        denominator = term1 + term2 + term3 + term4 + term5 + term6
        if denominator == 0:
           raise ValueError("计算θ时分母为零！请检查输入的动力学参数（k值是否全为零）。")
    
        # Calculate θ values (simplified)
        theta = {
            'theta*': term1 / denominator,
            'theta*OH': term2 / denominator,
            'theta*(OH)2': term3 / denominator,
            'theta*O': term4 / denominator,
            'theta*O(OH)': term5 / denominator,
            'theta*O(O)':term6 / denominator
        }
        
        return theta
     # 界面更新函数
    def update_results_table(self):
        self.tree.delete(*self.tree.get_children())
        columns = list(self.results_df.columns)
        self.tree["columns"] = columns
        
        # 设置列标题和初始宽度
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, 
                            width=tk.font.Font().measure(col[:15]) + 20,
                            anchor=tk.CENTER,
                            stretch=False)
        
        # 插入数据行
        for _, row in self.results_df.iterrows():
            formatted_row = [
                f"{x:.4e}" if isinstance(x, float) else str(x)
                for x in row
            ]
            self.tree.insert("", tk.END, values=list(row))
        
        # 动态调整列宽
        for col in columns:
            max_width = tk.font.Font().measure(col[:15])  # 限制标题显示长度
            for item in self.results_df[col]:
                item_str = f"{item:.4e}" if isinstance(item, float) else str(item)
                item_width = tk.font.Font().measure(item_str[:15])  # 限制内容显示长度
                if item_width > max_width:
                    max_width = item_width
            self.tree.column(col, width=max_width + 20)
                    

        # 添加滚动条自动隐藏功能（可选）
        def check_scrollbars():
            vsb_needed = self.tree.yview() != (0.0, 1.0)
            hsb_needed = self.tree.xview() != (0.0, 1.0)
    
            vsb.grid_configure(column=1, sticky="ns" if vsb_needed else "")
            hsb.grid_configure(row=1, sticky="ew" if hsb_needed else "")
    
            self.root.after(100, check_scrollbars)

    def update_plot(self):
        """优化后的绘图方法"""
        self.fig.set_size_inches(8, 6 + len(self.results_df)*0.1)  # 动态调整高度
        self.canvas.draw()
        try:
            # 清除旧数据
            [ax.clear() for ax in [self.ax_k, self.ax_lgr, self.ax_theta]]
            
            # 获取变量数据
            variable = self.results_df.columns[0]
            x = self.results_df[variable]
            
            # 设置统一颜色主题
            colors = plt.cm.tab10.colors
            
            # 绘制速率常数
            k_cols = [c for c in self.results_df.columns if c.startswith('k')]
            for idx, col in enumerate(k_cols):
                self.ax_k.plot(x, self.results_df[col], 
                              color=colors[idx%10],
                              linewidth=1.5,
                              label=col)
            self.ax_k.set_title("Rate Constants", fontsize=10, pad=8)
            self.ax_k.legend(fontsize=7, loc='upper right', framealpha=0.8)
            
            # 绘制对数速率
            lgr_cols = [c for c in self.results_df.columns if c.startswith('lg(r')]
            for idx, col in enumerate(lgr_cols):
                self.ax_lgr.plot(x, self.results_df[col], 
                                color=colors[idx%10],
                                linewidth=1.5,
                                label=col.replace('lg','log'))
            self.ax_lgr.set_title("Reaction Rates (log scale)", fontsize=10, pad=8)
            self.ax_lgr.legend(fontsize=7, loc='upper right', framealpha=0.8)
            
            # 绘制表面覆盖度
            theta_cols = [c for c in self.results_df.columns if c.startswith('theta')]
            for idx, col in enumerate(theta_cols):
                self.ax_theta.plot(x, self.results_df[col], 
                                  color=colors[idx%10],
                                  linewidth=1.5,
                                  label=col)
            self.ax_theta.set_title("Surface Coverage", fontsize=10, pad=8)
            self.ax_theta.legend(fontsize=7, loc='upper right', framealpha=0.8)
            
            # 统一设置坐标标签
            for ax in [self.ax_k, self.ax_lgr]:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            self.ax_theta.set_xlabel(variable, fontsize=9)
            
            # 优化布局
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("绘图错误", f"图表生成失败：\n{str(e)}")
            traceback.print_exc()

    # 保存功能
    def save_results(self):
        try:
            if hasattr(self, 'results_df'):
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
                    title="保存结果"
                )
                if file_path:
                    if file_path.endswith('.csv'):
                        self.results_df.to_csv(file_path, index=False)
                    else:
                        self.results_df.to_excel(file_path, index=False)
                    messagebox.showinfo("保存成功", f"文件已保存至：\n{file_path}")
            else:
                messagebox.showwarning("无数据", "请先进行计算")
        except Exception as e:
            messagebox.showerror("保存错误", f"保存失败：\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AOMKineticsGUI(root)
    root.mainloop()
