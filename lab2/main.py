import math
import random
import sys
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QCheckBox, QTextEdit, QComboBox, QInputDialog
from PyQt5.QtGui import QPainter, QPen, QBrush, QPolygonF
from PyQt5.QtCore import Qt, QPointF

cities = []
distance_matrix = np.array([])

def nearest_neighbor_tsp(matrix, start=0, optimization=False):
    n = len(matrix)
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    for _ in range(n - 1):
        last = path[-1]
        nearest = np.inf
        nearest_index = -1
        
        for i in range(n):
            if not visited[i] and matrix[last][i] > 0 and matrix[last][i] < nearest:
                nearest = matrix[last][i]
                nearest_index = i
        
        if nearest_index == -1:
            return [], 0
        
        path.append(nearest_index)
        visited[nearest_index] = True
    
    if matrix[path[-1]][start] > 0:
        path.append(start)
    else:
        return [], 0
    
    path_length = sum(matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))
    
    if optimization:
        best_path = path
        best_cost = path_length
        
        for new_start in range(n):
            if new_start == start:
                continue
            new_path, new_cost = nearest_neighbor_tsp(matrix, start=new_start)
            if new_path and new_cost < best_cost:
                best_path, best_cost = new_path, new_cost
        
        return best_path, best_cost
    
    return path, path_length

def parse_graph_input(graph_text):
    global cities, distance_matrix
    lines = graph_text.strip().split("\n")
    cities = lines[0].split()
    n = len(cities)
    distance_matrix = np.zeros((n, n))
    
    for i in range(1, len(lines)):
        values = list(map(int, lines[i].split()))
        for j in range(n):
            distance_matrix[i - 1][j] = values[j]

import random
import math

def nearest_neighbor(matrix, start):
    """
    Build initial path using the nearest neighbor heuristic.
    matrix: adjacency matrix where 0 indicates no edge.
    start: starting node index.
    Returns a list of nodes (without returning to start).
    """
    n = len(matrix)
    visited = {start}
    path = [start]
    current = start

    while len(visited) < n:
        next_node = None
        min_w = float('inf')
        for j in range(n):
            w = matrix[current][j]
            if w and j not in visited and w < min_w:
                min_w = w
                next_node = j
        if next_node is None:
            break  # no further moves
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path

def path_cost(matrix, path, start):
    """
    Compute total cost of a cycle defined by `path` returning to `start`.
    Returns float('inf') if any required edge is missing.
    """
    cost = 0
    # cost along the path
    for u, v in zip(path, path[1:]):
        w = matrix[u][v]
        if w == 0:
            return float('inf')
        cost += w
    # return to start
    w_back = matrix[path[-1]][start]
    if w_back == 0:
        return float('inf')
    cost += w_back
    return cost

def simulated_annealing_tsp(matrix, start=4, cauchy_modification=False, T=1000.0, alpha=0.99):
    """
    Solve TSP on a directed weighted graph (0 means no edge) using simulated annealing.
    
    Parameters:
        matrix (list of lists or 2D array): adjacency matrix
        start (int): index of starting node
        cauchy_modification (bool): if True, use Cauchy cooling schedule T = T/(1+0.01*T);
                                     otherwise use geometric schedule T *= alpha
        T (float): initial temperature
        alpha (float): cooling rate for geometric schedule
    
    Returns:
        best_cycle (list): sequence of nodes including return to start
        best_cost (float): total cost of the best cycle
    """
    # Initial solution via nearest neighbor
    current_path = nearest_neighbor(matrix, start)
    current_cost = path_cost(matrix, current_path, start)
    best_path = current_path.copy()
    best_cost = current_cost

    # Main loop: cool until temperature is very low
    while T > 1e-3 and len(current_path) > 1:
        # Generate a neighbor by reversing a random subsequence (excluding the first node)
        i, j = sorted(random.sample(range(1, len(current_path)), 2))
        new_path = current_path.copy()
        new_path[i:j] = list(reversed(new_path[i:j]))
        
        new_cost = path_cost(matrix, new_path, start)
        delta = new_cost - current_cost
        
        # Accept new solution if better, or with probability exp(-delta/T)
        if delta < 0 or math.exp(-delta / T) > random.random():
            current_path, current_cost = new_path, new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = current_path.copy()
        
        # Update temperature
        if cauchy_modification:
            T = T / (1 + 0.01 * T)
        else:
            T *= alpha

    # Build full cycle by returning to start
    best_cycle = best_path + [start]
    return best_cycle, best_cost

def acceptance_probability(delta, T, cauchy_modification):
    if cauchy_modification:
        return 1.0 / (1 + delta / T)
    else:
        return math.exp(-delta / T)

def draw_graph(path):
    if not path:
        print("Невозможно найти путь, возвращающийся в начальный город.")
        return
    
    G = nx.DiGraph()
    for i in range(len(cities)):
        for j in range(len(cities)):
            if distance_matrix[i][j] > 0:
                G.add_edge(cities[i], cities[j], weight=distance_matrix[i][j])
    
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    path_edges = [(cities[path[i]], cities[path[i + 1]]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, arrows=True)
    plt.show()

class GraphEditor(QWidget):
    def __init__(self, tsp_app):
        super().__init__()
        self.setWindowTitle("Редактор графа")
        self.resize(500, 500)
        self.nodes = []  # Список вершин
        self.edges = []  # Список рёбер (начальная вершина, конечная вершина, вес)
        self.tsp_app = tsp_app
        self.start_node = None  # Временное хранилище для соединения вершин

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            for i, (x, y) in enumerate(self.nodes):
                if (x - pos.x()) ** 2 + (y - pos.y()) ** 2 <= 400:  # Проверка нажатия на вершину
                    if self.start_node is None:
                        self.start_node = i  # Выбираем начальную вершину
                    else:
                        weight, ok = QInputDialog.getInt(self, "Введите вес", "Вес ребра:", min=1)
                        if ok:
                            self.edges.append((self.start_node, i, weight))
                        self.start_node = None  # Сброс выбора
                    self.update()
                    return
            
            # Добавляем новую вершину
            self.nodes.append((pos.x(), pos.y()))
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Рисуем рёбра со стрелками
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)
        for start, end, weight in self.edges:
            x1, y1 = self.nodes[start]
            x2, y2 = self.nodes[end]
            painter.drawLine(x1, y1, x2, y2)
            self.draw_arrow(painter, x1, y1, x2, y2)
            painter.drawText(int((x1 + x2) / 2), int((y1 + y2) / 2), str(weight))
        
        # Рисуем вершины
        painter.setBrush(QBrush(Qt.blue))
        for i, (x, y) in enumerate(self.nodes):
            painter.drawEllipse(QPointF(x, y), 10, 10)
            painter.drawText(int(x - 10), int(y - 10), str(i))

    def draw_arrow(self, painter, x1, y1, x2, y2):
        arrow_size = 20
        angle = np.arctan2(y2 - y1, x2 - x1)
        
        line_pen = QPen(Qt.black, 2)
        painter.setPen(line_pen)
        painter.drawLine(x1, y1, x2, y2)
        
        p1 = QPointF(x2 - arrow_size * np.cos(angle - np.pi / 6), y2 - arrow_size * np.sin(angle - np.pi / 6))
        p2 = QPointF(x2, y2)
        p3 = QPointF(x2 - arrow_size * np.cos(angle + np.pi / 6), y2 - arrow_size * np.sin(angle + np.pi / 6))
        
        arrow_head = QPolygonF([p1, p2, p3])
        painter.setBrush(QBrush(Qt.yellow))
        painter.drawPolygon(arrow_head)

    def get_graph_data(self):
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        for start, end, weight in self.edges:
            matrix[start][end] = weight
        return [str(i) for i in range(n)], matrix

class TSPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel("Введите граф (первая строка - города, далее матрица расстояний):")
        self.graph_input = QTextEdit()
        self.draw_graph_button = QPushButton("Создать граф вручную")
        self.draw_graph_button.clicked.connect(self.open_graph_editor)
        self.add_graph_button = QPushButton("Добавить граф")
        self.add_graph_button.clicked.connect(self.load_graph)
        self.optimization_checkbox = QCheckBox("Включить модификацию Коши")

        self.start_node_label = QLabel("Выберите начальный город:")
        self.start_node_selector = QComboBox()

        # Добавляем поля для гиперпараметров
        self.initial_temp_label = QLabel("Начальная температура:")
        self.initial_temp_input = QTextEdit("1000.0")
        self.initial_temp_input.setFixedHeight(30)

        self.cooling_rate_label = QLabel("Коэффициент охлаждения (0 < alpha < 1):")
        self.cooling_rate_input = QTextEdit("0.99")
        self.cooling_rate_input.setFixedHeight(30)

        self.button = QPushButton("Найти путь")
        self.button.clicked.connect(self.solve_tsp)

        self.result_label = QLabel("")
        self.path_label = QLabel("")
        self.path_length_label = QLabel("")
        
        layout.addWidget(self.label)
        layout.addWidget(self.graph_input)
        layout.addWidget(self.draw_graph_button)
        layout.addWidget(self.add_graph_button)
        layout.addWidget(self.start_node_label)
        layout.addWidget(self.start_node_selector)
        layout.addWidget(self.optimization_checkbox)

        layout.addWidget(self.initial_temp_label)
        layout.addWidget(self.initial_temp_input)
        layout.addWidget(self.cooling_rate_label)
        layout.addWidget(self.cooling_rate_input)

        layout.addWidget(self.button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.path_label)
        layout.addWidget(self.path_length_label)
        self.setLayout(layout)
    
    def load_graph(self):
        global cities, distance_matrix
        graph_text = self.graph_input.toPlainText()
        parse_graph_input(graph_text)
        if hasattr(self, 'graph_editor') and self.graph_editor.nodes:
            cities, distance_matrix = self.graph_editor.get_graph_data()
        self.start_node_selector.clear()
        self.start_node_selector.addItems(cities)
    
    def open_graph_editor(self):
        self.graph_editor = GraphEditor(self)
        self.graph_editor.show()

    def load_graph_from_file(self, filepath):
        return np.loadtxt(filepath, dtype=int)

    def test_algorithm(self):
        node_sizes = [6, 10, 15, 30, 100]
        num_runs = 5
        results = []

        for n in node_sizes:
            filepath = f"sparse_graph_{n}_nodes.txt"
            matrix = self.load_graph_from_file(filepath)

            costs_normal, times_normal = [], []
            costs_cauchy, times_cauchy = [], []

            for _ in range(num_runs):
                start_time = time.time()
                _, cost_normal = simulated_annealing_tsp(matrix, cauchy_modification=False)
                times_normal.append(time.time() - start_time)
                costs_normal.append(cost_normal)

                start_time = time.time()
                _, cost_cauchy = simulated_annealing_tsp(matrix, cauchy_modification=True)
                times_cauchy.append(time.time() - start_time)
                costs_cauchy.append(cost_cauchy)

            avg_cost_normal = np.mean(costs_normal)
            avg_time_normal = np.mean(times_normal)
            avg_cost_cauchy = np.mean(costs_cauchy)
            avg_time_cauchy = np.mean(times_cauchy)

            results.append((n, avg_cost_normal, avg_time_normal, avg_cost_cauchy, avg_time_cauchy))

        print("\nРазмер графа | Стоимость (обычный) | Время (обычный) | Стоимость (Коши) | Время (Коши)")
        for res in results:
            print(f"{res[0]:<12} | {res[1]:<20.2f} | {res[2]:<15.5f} | {res[3]:<18.2f} | {res[4]:<15.5f}")
    
    def solve_tsp(self):
        use_optimization = self.optimization_checkbox.isChecked()
        start_index = self.start_node_selector.currentIndex()

        try:
            initial_temp = float(self.initial_temp_input.toPlainText())
            cooling_rate = float(self.cooling_rate_input.toPlainText())
            if not (0 < cooling_rate < 1):
                raise ValueError("Коэффициент охлаждения должен быть от 0 до 1")
        except ValueError as e:
            self.result_label.setText(f"Ошибка ввода гиперпараметров: {e}")
            return
        self.test_algorithm()
        # path, path_length = simulated_annealing_tsp(
        #     distance_matrix, 
        #     start=start_index, 
        #     cauchy_modification=use_optimization,
        #     T=initial_temp,
        #     alpha=cooling_rate
        # )

        # if path:
        #     formatted_path = ' → '.join([cities[i] for i in path])
        #     self.result_label.setText(f"Оптимальный путь: {formatted_path}")
        #     self.path_length_label.setText(f"Длина пути: {path_length}")
        #     draw_graph(path)
        # else:
        #     self.result_label.setText("Невозможно построить цикл, возвращающийся в начальный город.")
        #     self.path_label.setText("")
        #     self.path_length_label.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TSPApp()
    ex.setWindowTitle("Задача коммивояжёра (Ориентированный граф)")

    ex.show()
    sys.exit(app.exec_())