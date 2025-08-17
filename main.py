import argparse
import os
import numpy as np
from network import NeuralNetwork
import struct
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
import threading
import time


def load_mnist(path="data"):
    def read_idx(filename):
        with open(filename, "rb") as f:
            data = f.read()
            magic, num_items = struct.unpack(">II", data[:8])
            if magic == 2051:
                rows, cols = struct.unpack(">II", data[8:16])
                images = np.frombuffer(data, dtype=np.uint8, offset=16)
                return images.reshape(num_items, rows * cols) / 255.0
            elif magic == 2049:
                return np.frombuffer(data, dtype=np.uint8, offset=8)
            else:
                raise ValueError(f"Invalid MNIST file: {filename}")

    x_train = read_idx(os.path.join(path, "train-images.idx3-ubyte"))
    y_train = read_idx(os.path.join(path, "train-labels.idx1-ubyte"))
    x_test = read_idx(os.path.join(path, "t10k-images.idx3-ubyte"))
    y_test = read_idx(os.path.join(path, "t10k-labels.idx1-ubyte"))

    x_train = [x.reshape(784, 1) for x in x_train]
    x_test = [x.reshape(784, 1) for x in x_test]

    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    training_data = list(zip(x_train, [vectorized_result(y) for y in y_train]))
    test_data = list(zip(x_test, y_test))

    return training_data, test_data


class DigitDrawer:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("MNIST Neural Network - Live Digit Recognition")
        self.window.geometry("1200x800")
        self.window.configure(bg='#1e1e1e')
        
        self.drawing_size = 400
        self.canvas_image = Image.new('L', (self.drawing_size, self.drawing_size), 255)
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        
        self.setup_ui()
        self.prediction_updating = False
        self.last_prediction_time = 0
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TFrame', background='#1e1e1e')
        style.configure('Title.TLabel', background='#1e1e1e', foreground='#ffffff', font=('Arial', 24, 'bold'))
        style.configure('Stat.TLabel', background='#1e1e1e', foreground='#00ff88', font=('Arial', 14, 'bold'))
        style.configure('Prob.TLabel', background='#2a2a2a', foreground='#ffffff', font=('Arial', 11))
        
        main_container = ttk.Frame(self.window, style='Custom.TFrame')
        main_container.pack(fill='both', expand=True, padx=30, pady=20)
        
        title = ttk.Label(main_container, text="Neural Network Live Digit Recognition", style='Title.TLabel')
        title.pack(pady=(0, 30))
        
        content_frame = ttk.Frame(main_container, style='Custom.TFrame')
        content_frame.pack(fill='both', expand=True)
        
        left_panel = ttk.Frame(content_frame, style='Custom.TFrame')
        left_panel.pack(side='left', fill='both', padx=(0, 40))
        
        canvas_frame = tk.Frame(left_panel, bg='#2a2a2a', relief='raised', bd=3)
        canvas_frame.pack(pady=(0, 20))
        
        self.canvas = tk.Canvas(canvas_frame, width=self.drawing_size, height=self.drawing_size, 
                               bg='#ffffff', cursor='crosshair', highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)
        
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.on_draw_end)
        
        button_frame = ttk.Frame(left_panel, style='Custom.TFrame')
        button_frame.pack()
        
        clear_btn = tk.Button(button_frame, text='Clear Canvas', command=self.clear_canvas,
                             font=('Arial', 14, 'bold'), bg='#ff4444', fg='white', 
                             relief='flat', padx=30, pady=10, cursor='hand2')
        clear_btn.pack()
        
        right_panel = ttk.Frame(content_frame, style='Custom.TFrame')
        right_panel.pack(side='right', fill='both')
        
        digits_frame = tk.Frame(right_panel, bg='#1e1e1e')
        digits_frame.pack(fill='x', pady=(0, 30))
        
        digits_label = ttk.Label(digits_frame, text="Confidence Levels", 
                                style='Title.TLabel', font=('Arial', 18, 'bold'))
        digits_label.pack(pady=(0, 15))
        
        self.digit_boxes = []
        boxes_container = tk.Frame(digits_frame, bg='#1e1e1e')
        boxes_container.pack()
        
        for i in range(10):
            digit_frame = tk.Frame(boxes_container, bg='#1e1e1e')
            digit_frame.grid(row=i//5, column=i%5, padx=8, pady=8)
            
            digit_label = tk.Label(digit_frame, text=str(i), font=('Arial', 16, 'bold'),
                                  bg='#2a2a2a', fg='#ffffff', width=3, height=2)
            digit_label.pack()
            
            confidence_bar = tk.Canvas(digit_frame, width=60, height=8, bg='#2a2a2a', 
                                     highlightthickness=0)
            confidence_bar.pack(pady=(5, 0))
            
            confidence_text = tk.Label(digit_frame, text='0.0%', font=('Arial', 10),
                                     bg='#1e1e1e', fg='#888888')
            confidence_text.pack()
            
            self.digit_boxes.append({
                'label': digit_label,
                'bar': confidence_bar,
                'text': confidence_text,
                'value': 0.0
            })
        
        stats_frame = tk.Frame(right_panel, bg='#2a2a2a', relief='raised', bd=2)
        stats_frame.pack(fill='x', pady=(0, 20))
        
        stats_title = tk.Label(stats_frame, text='Live Statistics', font=('Arial', 16, 'bold'),
                              bg='#2a2a2a', fg='#ffffff')
        stats_title.pack(pady=10)
        
        self.prediction_label = tk.Label(stats_frame, text='Prediction: —', 
                                        font=('Arial', 20, 'bold'), bg='#2a2a2a', fg='#00ff88')
        self.prediction_label.pack(pady=5)
        
        self.confidence_label = tk.Label(stats_frame, text='Max Confidence: 0.0%',
                                        font=('Arial', 14), bg='#2a2a2a', fg='#ffffff')
        self.confidence_label.pack(pady=3)
        
        self.entropy_label = tk.Label(stats_frame, text='Uncertainty: 0.000',
                                     font=('Arial', 14), bg='#2a2a2a', fg='#ffffff')
        self.entropy_label.pack(pady=3)
        
        self.loss_label = tk.Label(stats_frame, text='Loss: 0.0000',
                                  font=('Arial', 14), bg='#2a2a2a', fg='#ffffff')
        self.loss_label.pack(pady=(3, 10))

    def paint(self, event):
        x, y = event.x, event.y
        brush_size = 20
        
        self.canvas.create_oval(x - brush_size//2, y - brush_size//2,
                               x + brush_size//2, y + brush_size//2,
                               fill='black', outline='black')
        
        scale_x = self.canvas_image.width / self.drawing_size
        scale_y = self.canvas_image.height / self.drawing_size
        
        img_x = int(x * scale_x)
        img_y = int(y * scale_y)
        img_brush = int(brush_size * scale_x)
        
        self.canvas_draw.ellipse([img_x - img_brush//2, img_y - img_brush//2,
                                 img_x + img_brush//2, img_y + img_brush//2], fill=0)
        
        current_time = time.time()
        if current_time - self.last_prediction_time > 0.05:
            self.update_prediction_async()
            self.last_prediction_time = current_time

    def on_draw_end(self, event):
        self.update_prediction_async()

    def update_prediction_async(self):
        if not self.prediction_updating:
            self.prediction_updating = True
            threading.Thread(target=self.update_prediction, daemon=True).start()

    def update_prediction(self):
        try:
            img_resized = self.canvas_image.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_array = 1.0 - img_array
            input_data = img_array.flatten().reshape(784, 1)
            
            output = self.model.feedforward(input_data)
            probabilities = output.flatten()
            
            prediction = np.argmax(probabilities)
            max_confidence = probabilities[prediction]
            
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            
            target = np.zeros((10, 1))
            target[prediction] = 1.0
            loss = float(np.mean((output - target) ** 2))
            
            self.window.after(0, lambda: self.update_ui(probabilities, prediction, 
                                                       max_confidence, entropy, loss))
        finally:
            self.prediction_updating = False

    def update_ui(self, probabilities, prediction, max_confidence, entropy, loss):
        self.prediction_label.config(text=f'Prediction: {prediction}')
        self.confidence_label.config(text=f'Max Confidence: {max_confidence:.1%}')
        self.entropy_label.config(text=f'Uncertainty: {entropy:.3f}')
        self.loss_label.config(text=f'Loss: {loss:.4f}')
        
        for i, box in enumerate(self.digit_boxes):
            confidence = probabilities[i]
            box['value'] = confidence
            
            intensity = min(255, int(confidence * 255))
            if i == prediction:
                color = f'#{255-intensity:02x}{255:02x}{255-intensity:02x}'
                text_color = '#00ff00' if confidence > 0.5 else '#ffffff'
            else:
                color = f'#{255-intensity:02x}{255-intensity:02x}{255-intensity:02x}'
                text_color = '#ffffff' if confidence > 0.3 else '#888888'
            
            box['label'].config(bg=color, fg=text_color)
            box['text'].config(text=f'{confidence:.1%}', 
                             fg='#00ff00' if i == prediction else '#888888')
            
            box['bar'].delete('all')
            bar_width = int(60 * confidence)
            if bar_width > 0:
                bar_color = '#00ff88' if i == prediction else '#666666'
                box['bar'].create_rectangle(0, 0, bar_width, 8, fill=bar_color, outline='')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.canvas_image = Image.new('L', (self.drawing_size, self.drawing_size), 255)
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        
        self.prediction_label.config(text='Prediction: —')
        self.confidence_label.config(text='Max Confidence: 0.0%')
        self.entropy_label.config(text='Uncertainty: 0.000')
        self.loss_label.config(text='Loss: 0.0000')
        
        for box in self.digit_boxes:
            box['label'].config(bg='#2a2a2a', fg='#ffffff')
            box['text'].config(text='0.0%', fg='#888888')
            box['bar'].delete('all')

    def run(self):
        self.window.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the network")
    parser.add_argument("--test", action="store_true", help="Test the network accuracy")
    parser.add_argument("--model", type=str, default="mnist_final.npz", help="Path to save/load model")
    args = parser.parse_args()

    if args.train:
        training_data, test_data = load_mnist()
        nn = NeuralNetwork([784, 128, 64, 10])
        nn.train(training_data, epochs=30, mini_batch_size=32, eta=0.25,
                 test_data=test_data, track_cost=True)
        nn.save(args.model)
    else:
        if os.path.exists(args.model):
            nn = NeuralNetwork.load(args.model)
        else:
            print(f"Model file {args.model} not found. Please train first with --train")
            return

    if args.test:
        _, test_data = load_mnist()
        correct = nn.evaluate(test_data)
        total = len(test_data)
        accuracy = correct / total * 100
        print(f"📊 Accuracy of the trained model: {accuracy:.2f}% ({correct}/{total})")
    else:
        drawer = DigitDrawer(nn)
        drawer.run()


if __name__ == "__main__":
    main()