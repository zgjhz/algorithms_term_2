# A B C D E
# 0 3 1 4 8
# 2 0 5 1 3
# 6 2 0 3 7
# 4 3 8 0 2
# 5 6 3 4 0

import time
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QCheckBox, QTextEdit

cities = []
distance_matrix = np.array([])
has_cycle = True

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
            break
        
        path.append(nearest_index)
        visited[nearest_index] = True
    
    if matrix[path[-1]][start] <= 0:
        path.append(start)
        has_cycle = False

    
    if optimization:
        best_path = path
        best_cost = sum(matrix[best_path[i]][best_path[i + 1]] for i in range(len(path) - 1))
        
        for new_start in range(n):
            if new_start == start:
                continue
            new_path = nearest_neighbor_tsp(matrix, start=new_start)
            new_cost = sum(matrix[new_path[i]][new_path[i + 1]] for i in range(len(new_path) - 1))
            if new_cost < best_cost:
                best_path, best_cost = new_path, new_cost
        
        return best_path
    
    return path

def draw_graph(path):
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

class TSPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel("Введите граф (первая строка - города, далее матрица расстояний):")
        self.graph_input = QTextEdit()
        self.optimization_checkbox = QCheckBox("Включить модификацию")
        self.button = QPushButton("Найти путь")
        self.button.clicked.connect(self.solve_tsp)
        self.result_label = QLabel("")
        self.path_label = QLabel("")
        
        layout.addWidget(self.label)
        layout.addWidget(self.graph_input)
        layout.addWidget(self.optimization_checkbox)
        layout.addWidget(self.button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.path_label)
        self.setLayout(layout)
    
    def solve_tsp(self):
        graph_text = self.graph_input.toPlainText()
        parse_graph_input(graph_text)
        use_optimization = self.optimization_checkbox.isChecked()
        time_s = time.time()
        path = nearest_neighbor_tsp(distance_matrix, optimization=use_optimization)
        print(time.time() - time_s)
        if has_cycle:
            formatted_path = ' → '.join([cities[i] for i in path])
        else:
            formatted_path = "Нет такого пути"
        self.result_label.setText(f"Оптимальный путь: {formatted_path}")
        # self.path_label.setText(f"Путь: {' → '.join([cities[i] for i in path])}")
        draw_graph(path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TSPApp()
    ex.setWindowTitle("Задача коммивояжёра (Ориентированный граф)")
    ex.resize(400, 400)
    ex.show()
    sys.exit(app.exec_())
