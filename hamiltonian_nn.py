import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + attended)
        fed_forward = self.feed_forward(x)
        return self.norm2(x + fed_forward)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64, num_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class HamiltonianDynamics:
    def __init__(self, mass: float = 1.0, dt: float = 0.01, friction: float = 0.1):
        self.mass = mass
        self.dt = dt
        self.friction = friction
        self.initial_energy = None
        # Convert scalars to tensors
        self.friction_tensor = torch.tensor(friction)
        self.dt_tensor = torch.tensor(dt)
        self.mass_tensor = torch.tensor(mass)
        
    def compute_hamiltonian(self, q: torch.Tensor, p: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        T = torch.sum(p**2) / (2 * self.mass)
        return T + V
        
    def leapfrog_step(self, q: torch.Tensor, p: torch.Tensor, 
                      grad_V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Move tensors to same device as input
        device = q.device
        self.friction_tensor = self.friction_tensor.to(device)
        self.dt_tensor = self.dt_tensor.to(device)
        self.mass_tensor = self.mass_tensor.to(device)
        
        H = self.compute_hamiltonian(q, p, torch.sum(grad_V**2))
        
        if self.initial_energy is None:
            self.initial_energy = H.item()
        
        # Modified leapfrog with tensor operations
        p = p - 0.5 * self.dt_tensor * grad_V
        friction_factor = torch.exp(-self.friction_tensor * self.dt_tensor)
        p = p * friction_factor
        q = q + self.dt_tensor * p / self.mass_tensor
        p = p - 0.5 * self.dt_tensor * grad_V
        
        H_current = self.compute_hamiltonian(q, p, torch.sum(grad_V**2))
        scale_factor = torch.sqrt(torch.tensor(self.initial_energy, device=device) / H_current)
        p = p * scale_factor
        
        return q, p

class HamiltonianLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module = nn.CrossEntropyLoss(),
                 mass: float = 1.0, dt: float = 0.01, friction: float = 0.1):
        super().__init__()
        self.base_criterion = base_criterion
        self.dynamics = HamiltonianDynamics(mass=mass, dt=dt, friction=friction)
        self.momentum = None
        self.prev_params = None
        
    def init_momentum(self, params: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(params) * 0.01
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, 
                model: nn.Module) -> Tuple[torch.Tensor, Dict]:
        # Compute loss (potential energy)
        V = self.base_criterion(outputs, targets)
        
        # Get current parameters
        current_params = torch.cat([p.view(-1) for p in model.parameters()])
        
        if self.momentum is None:
            self.momentum = self.init_momentum(current_params)
            self.prev_params = current_params.clone().detach()
            return V, {"hamiltonian": V.item(), "kinetic": 0.0, "potential": V.item()}
        
        # Compute gradients
        grad_V = torch.autograd.grad(V, model.parameters(), create_graph=True)
        grad_V = torch.cat([g.view(-1) for g in grad_V])
        
        # Update parameters and momentum using Hamiltonian dynamics
        new_params, new_momentum = self.dynamics.leapfrog_step(
            current_params, self.momentum, grad_V
        )
        
        # Compute energies
        T = torch.sum(new_momentum**2) / (2 * self.dynamics.mass)
        H = self.dynamics.compute_hamiltonian(new_params, new_momentum, V)
        
        # Update state
        self.momentum = new_momentum.detach()
        self.prev_params = current_params.detach()
        
        # Return metrics
        metrics = {
            "hamiltonian": H.item(),
            "kinetic": T.item(),
            "potential": V.item()
        }
        
        return H, metrics

class SimpleDataset(Dataset):
    def __init__(self, texts: list, labels: list, vocab_size: int = 1000):
        self.vocab = {}
        self.texts = []
        for text in texts:
            tokens = []
            for word in text.lower().split():
                if word not in self.vocab and len(self.vocab) < vocab_size - 1:
                    self.vocab[word] = len(self.vocab) + 1
                tokens.append(self.vocab.get(word, 0))
            self.texts.append(torch.tensor(tokens))
        self.labels = torch.tensor(labels)
        
    def __getitem__(self, idx: int) -> Dict:
        return {
            'input_ids': self.texts[idx],
            'labels': self.labels[idx]
        }

    def __len__(self) -> int:
        return len(self.labels)

def collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    input_ids = torch.stack([
        torch.nn.functional.pad(item['input_ids'], (0, max_len - len(item['input_ids'])))
        for item in batch
    ])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels}

class TrainingLogger:
    def __init__(self):
        self.metrics_history = []
        
    def log_epoch(self, epoch: int, total_epochs: int, metrics: dict):
        metrics_with_epoch = {'epoch': epoch + 1, **metrics}
        self.metrics_history.append(metrics_with_epoch)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row(
            "Epoch Progress", 
            f"{epoch + 1}/{total_epochs} ({((epoch + 1)/total_epochs)*100:.1f}%)"
        )
        
        for metric, value in metrics.items():
            table.add_row(metric.capitalize(), f"{value:.4f}")
        
        energy_ratio = metrics['kinetic'] / metrics['potential'] if metrics['potential'] != 0 else float('inf')
        table.add_row("T/V Ratio", f"{energy_ratio:.4f}")
        
        console.print("\n")
        console.print(table)
        
    def save_metrics(self):
        df = pd.DataFrame(self.metrics_history)
        df['T/V Ratio'] = df['kinetic'] / df['potential']
        df.to_csv('training_metrics.csv', index=False)
        
        self.plot_training_history()
        self.plot_phase_space()
        
    def plot_training_history(self):
        df = pd.DataFrame(self.metrics_history)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        sns.lineplot(data=df, x='epoch', y='hamiltonian', label='Total Energy (H)', ax=ax1)
        sns.lineplot(data=df, x='epoch', y='kinetic', label='Kinetic Energy (T)', ax=ax1)
        sns.lineplot(data=df, x='epoch', y='potential', label='Potential Energy (V)', ax=ax1)
        ax1.set_title('Energy Components Over Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Energy')
        
        df['T/V Ratio'] = df['kinetic'] / df['potential']
        sns.lineplot(data=df, x='epoch', y='T/V Ratio', ax=ax2)
        ax2.set_title('Kinetic/Potential Energy Ratio Over Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('T/V Ratio')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
    def plot_phase_space(self):
        df = pd.DataFrame(self.metrics_history)
        plt.figure(figsize=(10, 6))
        plt.scatter(df['kinetic'], df['potential'], c=df['epoch'], cmap='viridis')
        plt.colorbar(label='Epoch')
        plt.xlabel('Kinetic Energy (T)')
        plt.ylabel('Potential Energy (V)')
        plt.title('Phase Space Trajectory')
        plt.savefig('phase_space.png')
        plt.close()
    def plot_energy_landscape(self):
        """
        Creates multiple visualizations of the energy landscape and system trajectory
        """
        df = pd.DataFrame(self.metrics_history)
        df['T/V Ratio'] = df['kinetic'] / df['potential']
        
        # Main figure with 4 subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 3D Phase Space Trajectory
        ax1 = fig.add_subplot(221, projection='3d')
        scatter = ax1.scatter(df['kinetic'], 
                            df['potential'], 
                            df['hamiltonian'],
                            c=df.index, 
                            cmap='viridis',
                            s=50)
        ax1.set_xlabel('Kinetic Energy (T)')
        ax1.set_ylabel('Potential Energy (V)')
        ax1.set_zlabel('Total Energy (H)')
        ax1.set_title('3D Energy Phase Space')
        fig.colorbar(scatter, label='Training Epoch')
        
        # 2. Energy Components Over Time
        ax2 = fig.add_subplot(222)
        ax2.plot(df['epoch'], df['hamiltonian'], label='Total (H)', linewidth=2)
        ax2.plot(df['epoch'], df['kinetic'], label='Kinetic (T)', linewidth=2)
        ax2.plot(df['epoch'], df['potential'], label='Potential (V)', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy Components Over Time')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)
        
        # 3. T/V Ratio Phase Plot
        ax3 = fig.add_subplot(223)
        points = ax3.scatter(df['kinetic'], df['potential'], 
                            c=df.index, cmap='viridis', s=50)
        ax3.set_xlabel('Kinetic Energy (T)')
        ax3.set_ylabel('Potential Energy (V)')
        ax3.set_title('T-V Phase Space')
        
        # Add trajectory arrows
        arrow_scale = max(df['kinetic'].max(), df['potential'].max())/100
        for i in range(len(df)-1):
            ax3.arrow(df['kinetic'].iloc[i], df['potential'].iloc[i],
                    df['kinetic'].iloc[i+1] - df['kinetic'].iloc[i],
                    df['potential'].iloc[i+1] - df['potential'].iloc[i],
                    head_width=arrow_scale,
                    head_length=arrow_scale,
                    fc='gray', ec='gray', alpha=0.5)
        
        fig.colorbar(points, label='Training Epoch')
        
        # Add constant energy contours
        T, V = np.meshgrid(np.linspace(0, df['kinetic'].max(), 100),
                        np.linspace(0, df['potential'].max(), 100))
        H = T + V
        ax3.contour(T, V, H, levels=10, colors='gray', alpha=0.3)
        
        # 4. Energy Ratio Over Time
        ax4 = fig.add_subplot(224)
        ax4.plot(df['epoch'], df['T/V Ratio'], color='purple', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('T/V Ratio')
        ax4.set_title('Energy Ratio Evolution')
        ax4.set_yscale('log')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('energy_landscape.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Separate Energy Flow Diagram
        plt.figure(figsize=(12, 6))
        
        # Create meshgrid for flow visualization
        T, V = np.meshgrid(np.linspace(0, df['kinetic'].max(), 100),
                        np.linspace(0, df['potential'].max(), 100))
        H = T + V
        
        # Calculate and normalize gradients
        dT = -np.gradient(H)[1]
        dV = -np.gradient(H)[0]
        magnitude = np.sqrt(dT**2 + dV**2)
        dT = np.where(magnitude > 0, dT/magnitude, 0)
        dV = np.where(magnitude > 0, dV/magnitude, 0)
        
        # Create streamplot with correct parameters
        plt.streamplot(T, V, dT, dV, color='gray', density=1.0, linewidth=0.5)
        
        # Add trajectory points
        plt.scatter(df['kinetic'], df['potential'], 
                c=df.index, cmap='viridis', s=50)
        plt.colorbar(label='Training Epoch')
        plt.xlabel('Kinetic Energy (T)')
        plt.ylabel('Potential Energy (V)')
        plt.title('Energy Flow Diagram')
        plt.savefig('energy_flow.png', dpi=300, bbox_inches='tight')
        plt.close()
class HamiltonianTrainer:
    def __init__(self, model: nn.Module, device: torch.device, config: Dict):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.loss_fn = HamiltonianLoss(
            mass=config.get('mass', 0.5),
            dt=config.get('dt', 0.001),
            friction=config.get('friction', 0.0)
        )
        self.logger = TrainingLogger()
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict:
        self.model.train()
        total_metrics = {"hamiltonian": 0.0, "kinetic": 0.0, "potential": 0.0}
        
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), 
                     TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("[cyan]Training...", total=len(dataloader))
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                loss, metrics = self.loss_fn(outputs, labels, self.model)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                for k, v in metrics.items():
                    total_metrics[k] += v
                    
                progress.advance(task)
                
        for k in total_metrics:
            total_metrics[k] /= len(dataloader)
            
        return total_metrics

def main():
    config = {
        'vocab_size': 1000,
        'embed_dim': 32,        # Reduced complexity
        'num_heads': 2,
        'num_layers': 1,
        'num_classes': 2,
        'batch_size': 4,
        'num_epochs': 1000,
        'learning_rate': 5e-5,  # Reduced learning rate
        'mass': 2.0,           # Increased mass for more inertia
        'dt': 0.0005,          # Smaller time step
        'friction': 0.2,       # Increased friction
        'grad_clip_threshold': 0.25,
        'warmup_steps': 50,    # Longer warmup
        'patience': 20,        # Early stopping patience
        'min_lr': 1e-6,       # Minimum learning rate
        'energy_threshold': 0.5 # Maximum allowed energy spike
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Using device: {device}")
    
    # 2. Expand the dataset
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps across a sleeping dog",
        "The cat and dog are friends",
        "The feline and canine are companions",
        "Dogs chase cats in the park",
        "Cats run away from dogs quickly",
        "Animals play together in harmony",
        "Pets enjoy their time together",
        "The dog barks at the mailman",
        "The cat sleeps on the windowsill",
        "Birds fly over the garden",
        "Fish swim in the pond",
    ] * 4  # 48 samples total

    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 4
    
    dataset = SimpleDataset(texts, labels, vocab_size=config['vocab_size'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    model = SimpleTransformer(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    )
    
    trainer = HamiltonianTrainer(model, device, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    console.print("[bold green]Starting training...\n")
    
    try:
        for epoch in range(config['num_epochs']):
            metrics = trainer.train_epoch(dataloader, optimizer)
            trainer.logger.log_epoch(epoch, config['num_epochs'], metrics)
        
        trainer.logger.save_metrics()
        trainer.logger.plot_energy_landscape()  # Add this line
        console.print("\n[bold green]Training completed! ✓")
        console.print("Training history plots and metrics saved.")
        
    except Exception as e:
        console.print(f"\n[bold red]Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()