import sys
import time

import openai
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread


class BenchmarkWorker(QThread):
    """Runs benchmark in a separate thread."""

    results_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(float)
    error_signal = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            client = openai.OpenAI(base_url=self.config["base_url"], api_key=self.config["api_key"])
            total_tps = []

            for i in range(self.config["runs"]):
                start = time.time()

                response = client.chat.completions.create(
                    model=self.config["model"],
                    messages=[{"role": "user", "content": self.config["prompt"]}],
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    stream=False
                )

                elapsed = time.time() - start
                completion_tokens = response.usage.completion_tokens
                tps = completion_tokens / elapsed
                total_tps.append(tps)

                self.results_signal.emit(
                    f"Run {i + 1}: {completion_tokens} tokens | "
                    f"{elapsed:.2f}s | ⚡ {tps:.2f} TPS"
                )

            avg_tps = sum(total_tps) / len(total_tps)
            self.finish_signal.emit(avg_tps)

        except Exception as e:
            self.error_signal.emit(str(e))


class BenchmarkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("OpenAI API Benchmark Tool")
        self.setMinimumSize(700, 650)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Connection settings group
        conn_group = QGroupBox("Connection Settings")
        conn_layout = QVBoxLayout()

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Base URL:"))
        self.base_url_input = QLineEdit("http://localhost:1234/v1")
        h1.addWidget(self.base_url_input)
        conn_layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit("no-api-key")
        h2.addWidget(self.api_key_input)
        conn_layout.addLayout(h2)

        conn_group.setLayout(conn_layout)
        main_layout.addWidget(conn_group)

        # Model selection group
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        model_layout.addWidget(self.model_combo, 1)

        self.refresh_models_btn = QPushButton("🔄 Refresh Models")
        self.refresh_models_btn.clicked.connect(self.load_models)
        model_layout.addWidget(self.refresh_models_btn)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # Benchmark settings group
        bench_group = QGroupBox("Benchmark Settings")
        bench_layout = QVBoxLayout()

        row_layout = QHBoxLayout()
        row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        row_layout.addWidget(QLabel("Max Tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 8192)
        self.max_tokens_spin.setValue(512)
        row_layout.addWidget(self.max_tokens_spin)

        row_layout.addSpacing(20)
        row_layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setValue(0.0)
        self.temp_spin.setSingleStep(0.1)
        row_layout.addWidget(self.temp_spin)

        row_layout.addSpacing(20)
        row_layout.addWidget(QLabel("Runs:"))
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 50)
        self.runs_spin.setValue(3)
        row_layout.addWidget(self.runs_spin)

        bench_layout.addLayout(row_layout)

        bench_group.setLayout(bench_layout)
        main_layout.addWidget(bench_group)

        # Prompt group
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()
        self.prompt_edit = QTextEdit(
            "Explain quantum computing in 1000 words. Include key concepts like superposition, entanglement, and qubits."
        )
        self.prompt_edit.setMaximumHeight(80)
        prompt_layout.addWidget(self.prompt_edit)
        prompt_group.setLayout(prompt_layout)
        main_layout.addWidget(prompt_group)

        # Buttons row
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run Benchmark")
        self.run_btn.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; font-size: 14px; padding: 8px; }
            QPushButton:hover { background-color: #218838; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        self.run_btn.clicked.connect(self.run_benchmark)
        btn_layout.addWidget(self.run_btn, 1)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton { background-color: #dc3545; color: white; font-size: 14px; padding: 8px; }
            QPushButton:hover { background-color: #c82333; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        btn_layout.addWidget(self.stop_btn)

        main_layout.addLayout(btn_layout)

        # Results area group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(200)
        results_layout.addWidget(self.output_text)

        # Summary table
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(4)
        self.summary_table.setHorizontalHeaderLabels(["Run", "Tokens", "Time (s)", "TPS"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setMaximumHeight(150)
        results_layout.addWidget(self.summary_table)

        h5 = QHBoxLayout()
        self.avg_label = QLabel("Average TPS: ---")
        self.avg_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #007bff;")
        h5.addWidget(self.avg_label, 1)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        # Load models on start
        self.load_models()

    def load_models(self):
        """Fetch available models from the OpenAI-compatible API server."""
        try:
            client = openai.OpenAI(
                base_url=self.base_url_input.text().strip(),
                api_key=self.api_key_input.text().strip()
            )
            models_response = client.models.list()
            model_ids = [m.id for m in models_response.data]

            current = self.model_combo.currentText()
            self.model_combo.clear()

            if not model_ids:
                self.model_combo.addItem("(no models loaded)")
                self.log("⚠️ No models found on the server. Make sure a model is loaded in your API server.")
            else:
                for mid in model_ids:
                    self.model_combo.addItem(mid)
                # Try to restore previously selected model
                if current and current in model_ids:
                    self.model_combo.setCurrentText(current)
                self.log(f"✅ Loaded {len(model_ids)} model(s): {', '.join(model_ids)}")

        except Exception as e:
            self.model_combo.clear()
            self.model_combo.addItem("(error loading models)")
            self.log(f"❌ Failed to load models: {e}")

    def run_benchmark(self):
        """Start benchmarking."""
        model = self.model_combo.currentText()
        if not model or model.startswith("("):
            QMessageBox.warning(self, "No Model", "Select a valid model from the list.")
            return

        config = {
            "base_url": self.base_url_input.text().strip(),
            "api_key": self.api_key_input.text().strip(),
            "model": model,
            "prompt": self.prompt_edit.toPlainText(),
            "max_tokens": self.max_tokens_spin.value(),
            "temperature": float(self.temp_spin.value()),
            "runs": self.runs_spin.value()
        }

        # Disable controls during run
        self.set_controls_enabled(False)
        self.output_text.clear()
        self.summary_table.setRowCount(0)
        self.avg_label.setText("Average TPS: ---")

        self.log(f"🧪 Benchmarking {model} ({config['runs']} runs, "
                 f"{config['max_tokens']} max tokens, temp={config['temperature']})\n")

        self.worker = BenchmarkWorker(config)
        self.worker.results_signal.connect(self.on_result)
        self.worker.finish_signal.connect(self.on_finish)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_result(self, msg):
        """Handle a single run result."""
        parts = msg.split(": ", 1)[1].split(" | ")
        run_num = int(msg.split(":")[0].replace("Run ", ""))
        tokens_str = "".join(ch for ch in parts[0].strip().split()[0] if ch.isdigit())
        time_str = "".join(ch for ch in parts[1].strip().split()[0] if ch.isdigit() or ch == ".")
        tps_str = "".join(ch for ch in parts[2].strip().split(" ")[1] if ch.isdigit() or ch == ".")

        tokens = int(tokens_str)
        time_val = float(time_str)
        tps = float(tps_str)

        row = self.summary_table.rowCount()
        self.summary_table.insertRow(row)
        self.summary_table.setItem(row, 0, QTableWidgetItem(str(run_num)))
        self.summary_table.setItem(row, 1, QTableWidgetItem(str(tokens)))
        self.summary_table.setItem(row, 2, QTableWidgetItem(f"{time_val:.2f}"))
        self.summary_table.setItem(row, 3, QTableWidgetItem(f"{tps:.2f}"))

        self.output_text.append(msg)

    def on_finish(self, avg_tps):
        """Handle benchmark completion."""
        self.set_controls_enabled(True)
        self.avg_label.setText(f"✅ Average TPS: {avg_tps:.2f}")
        self.output_text.append(f"\n✅ Benchmark complete. Average TPS: {avg_tps:.2f}")

    def on_error(self, msg):
        """Handle benchmark error."""
        self.set_controls_enabled(True)
        self.avg_label.setText("❌ Error")
        self.output_text.append(f"❌ Error: {msg}")
        QMessageBox.critical(self, "Benchmark Error", f"An error occurred:\n{msg}")

    def set_controls_enabled(self, enabled):
        """Enable/disable UI controls during benchmark."""
        self.base_url_input.setEnabled(enabled)
        self.api_key_input.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.refresh_models_btn.setEnabled(enabled)
        self.max_tokens_spin.setEnabled(enabled)
        self.temp_spin.setEnabled(enabled)
        self.runs_spin.setEnabled(enabled)
        self.prompt_edit.setEnabled(enabled)
        self.run_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)

    def log(self, msg):
        """Append message to output text."""
        self.output_text.append(msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BenchmarkApp()
    window.show()
    sys.exit(app.exec())