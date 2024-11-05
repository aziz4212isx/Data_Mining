import math
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx

class DecisionTreeNode:
    def __init__(self, data, attributes, depth=0, min_samples_split=2, max_depth=None):
        self.data = data
        self.children = {}
        self.split_attribute = None
        self.split_index = None
        self.split_value = None
        self.leaf_value = None
        self.depth = depth
        self.entropy = self._calculate_entropy(data)
        self.samples = len(data)
        self.value = self._calculate_class_distribution(data)
        
        class_values = [row[-1] for row in data]
        if (len(set(class_values)) == 1 or 
            len(data) < min_samples_split or 
            (max_depth is not None and depth >= max_depth)):
            self.leaf_value = Counter(class_values).most_common(1)[0][0]
            return
            
        gains = {}
        for idx, attribute in enumerate(attributes):
            if attribute is not None:
                gain = self._calculate_gain(data, idx)
                gains[idx] = gain
                
        if not gains:
            self.leaf_value = Counter(class_values).most_common(1)[0][0]
            return
            
        self.split_index = max(gains.items(), key=lambda x: x[1])[0]
        self.split_attribute = attributes[self.split_index]
        
        self.split_value = self._find_best_split(data, self.split_index)
        
        new_attributes = attributes.copy()
        new_attributes[self.split_index] = None
        
        left_data = [row for row in data if row[self.split_index] <= self.split_value]
        right_data = [row for row in data if row[self.split_index] > self.split_value]
        
        if left_data:
            self.children["True"] = DecisionTreeNode(left_data, new_attributes, depth + 1, 
                                                   min_samples_split, max_depth)
        if right_data:
            self.children["False"] = DecisionTreeNode(right_data, new_attributes, depth + 1, 
                                                    min_samples_split, max_depth)

    def _calculate_entropy(self, data):
        if not data:
            return 0
        
        total = len(data)
        class_counts = Counter(row[-1] for row in data)
        entropy = 0.0
        
        for count in class_counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
            
        return round(entropy, 3)

    def _calculate_class_distribution(self, data):
        if not data:
            return [0, 0]
        
        classes = ["Ya", "Tidak"]
        class_counts = Counter(row[-1] for row in data)
        return [class_counts[c] for c in classes]

    def _find_best_split(self, data, attribute_index):
        values = sorted(set(row[attribute_index] for row in data))
        if len(values) == 1:
            return values[0]
        
        best_gain = -1
        best_split = values[0]
        
        for i in range(len(values) - 1):
            split = values[i]
            left_data = [row for row in data if row[attribute_index] == split]
            right_data = [row for row in data if row[attribute_index] != split]
            
            gain = (len(left_data) / len(data) * self._calculate_entropy(left_data) +
                   len(right_data) / len(data) * self._calculate_entropy(right_data))
            
            if gain > best_gain:
                best_gain = gain
                best_split = split
        
        return best_split

    def _calculate_gain(self, data, attribute_index):
        parent_entropy = self._calculate_entropy(data)
        split_value = self._find_best_split(data, attribute_index)
        
        left_data = [row for row in data if row[attribute_index] == split_value]
        right_data = [row for row in data if row[attribute_index] != split_value]
        
        n = len(data)
        weighted_entropy = (len(left_data) / n * self._calculate_entropy(left_data) +
                            len(right_data) / n * self._calculate_entropy(right_data))
        
        return parent_entropy - weighted_entropy

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.root = None
        self.attributes = []
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, data, attributes):
        self.attributes = attributes
        self.root = DecisionTreeNode(data, attributes, 
                                     min_samples_split=self.min_samples_split,
                                     max_depth=self.max_depth)

    def visualize(self, filename="pohon_keputusan.jpg"):
        G = nx.DiGraph()
        pos = {}
        labels = {}
        colors = {}
        
        def add_nodes(node, x=0, y=0, dx=1):
            if node is None:
                return
                
            node_id = id(node)
            
            if node.leaf_value is not None:
                label = f'entropy = {node.entropy}\nsamples = {node.samples}\n'
                label += f'value = {node.value}\nclass = {node.leaf_value}'
                color = '#98FB98' if node.leaf_value == "Ya" else '#FF6347'
            else:
                label = f'{node.split_attribute} = {node.split_value}\n'
                label += f'entropy = {node.entropy}\nsamples = {node.samples}\n'
                label += f'value = {node.value}'
                color = 'white'
            
            G.add_node(node_id)
            pos[node_id] = (x, y)
            labels[node_id] = label
            colors[node_id] = color
            
            if "True" in node.children:
                child = node.children["True"]
                child_id = id(child)
                G.add_edge(node_id, child_id, label="True")
                add_nodes(child, x - dx, y - 1, dx / 2)
                
            if "False" in node.children:
                child = node.children["False"]
                child_id = id(child)
                G.add_edge(node_id, child_id, label="False")
                add_nodes(child, x + dx, y - 1, dx / 2)
        
        add_nodes(self.root)
        
        plt.figure(figsize=(15, 10))
        nx.draw(G, pos, labels=labels, node_color=[colors[node] for node in G.nodes()],
                with_labels=True, node_size=3000, node_shape='s',
                font_size=8, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.axis('off')
        plt.savefig(filename)
        plt.show()

def main():
    # Dataset keputusan membeli laptop
    data_laptop = [
        ["Tinggi", "Mahal", "Core i7", "16GB", "SSD", "Ya"],
        ["Rendah", "Murah", "Core i3", "4GB", "HDD", "Tidak"],
        ["Sedang", "Sedang", "Core i5", "8GB", "SSD", "Ya"],
        ["Tinggi", "Mahal", "Core i7", "8GB", "HDD", "Tidak"],
        ["Rendah", "Murah", "Core i3", "8GB", "SSD", "Ya"],
        ["Sedang", "Mahal", "Core i5", "16GB", "SSD", "Ya"],
        ["Tinggi", "Sedang", "Core i7", "8GB", "SSD", "Ya"],
        ["Rendah", "Sedang", "Core i3", "4GB", "HDD", "Tidak"],
        ["Sedang", "Murah", "Core i5", "8GB", "HDD", "Tidak"],
        ["Tinggi", "Mahal", "Core i7", "16GB", "HDD", "Ya"]
    ]
    
    atribut_laptop = ["Performa", "Harga", "Processor", "RAM", "Storage"]
    
    tree = DecisionTree(min_samples_split=2, max_depth=3)
    tree.fit(data_laptop, atribut_laptop)
    tree.visualize()

if __name__ == "__main__":
    main()
