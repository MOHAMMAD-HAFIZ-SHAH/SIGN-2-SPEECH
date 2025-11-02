"""
Comprehensive Visualization Script for CNN Model Performance
Generates graphs, charts, and confusion matrices for ASL Sign Language Model
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import json
from collections import Counter
import pandas as pd

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def safe_savefig(filepath, **kwargs):
    """Safely save figure with error handling"""
    try:
        plt.savefig(filepath, **kwargs)
        print(f"   Saved: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        print(f"   Error saving {os.path.basename(filepath)}: {e}")
        try:
            # Try saving with lower DPI if high DPI fails
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved with lower DPI: {os.path.basename(filepath)}")
            return True
        except:
            return False

# Configuration
IMG_SIZE = 64  # Model uses 64x64 images
MIN_THRESHOLD_VALUE = 70
DATASET_PATH = 'dataset'  # Dataset in model-training directory
MODEL_PATH = '../frontend/public/models/best_model.h5'  # Using existing trained model
METADATA_PATH = '../frontend/public/models/model_metadata.json'
OUTPUT_DIR = 'visualizations'

def preprocess_image_asl_style(img, img_size=IMG_SIZE, min_value=MIN_THRESHOLD_VALUE):
    """Same preprocessing as training"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        ret, res = cv2.threshold(
            th3, min_value, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        resized = cv2.resize(res, (img_size, img_size))
        return resized
    except Exception as e:
        print(f'Exception in preprocessing: {e}')
        return None

def load_test_data(data_path, img_size=IMG_SIZE, max_per_class=100):
    """Load test dataset with sampling for faster processing"""
    print(f"Loading test dataset (max {max_per_class} samples per class)...")
    
    categories = sorted([d for d in os.listdir(data_path) 
                        if os.path.isdir(os.path.join(data_path, d)) and len(d) == 1])
    
    label_dict = {category: idx for idx, category in enumerate(categories)}
    
    data = []
    target = []
    file_paths = []
    
    for category in categories:
        cat_path = os.path.join(data_path, category)
        img_names = [f for f in os.listdir(cat_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample images if there are too many
        if len(img_names) > max_per_class:
            img_names = np.random.choice(img_names, max_per_class, replace=False)
        
        print(f"  Processing {category}: {len(img_names)} images")
        
        for img_name in img_names:
            img_path = os.path.join(cat_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            processed = preprocess_image_asl_style(img, img_size)
            
            if processed is not None:
                data.append(processed)
                target.append(label_dict[category])
                file_paths.append(img_path)
    
    data = np.array(data) / 255.0
    data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
    target = np.array(target)
    
    print(f"\nTotal loaded: {len(data)} test images from {len(categories)} classes")
    return data, target, categories, label_dict, file_paths

def plot_dataset_distribution(target, categories, output_dir):
    """Plot distribution of samples per class"""
    print("\n1. Creating dataset distribution plot...")
    
    class_counts = Counter(target)
    labels = [categories[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    
    plt.figure(figsize=(16, 6))
    bars = plt.bar(labels, counts, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
    plt.xlabel('Sign Language Letter', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Dataset Distribution - Samples per Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0, fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_dataset_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 1_dataset_distribution.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, categories, output_dir):
    """Plot confusion matrix"""
    print("\n2. Creating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot both raw and normalized
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories,
                cbar_kws={'label': 'Count'}, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=categories, yticklabels=categories,
                cbar_kws={'label': 'Proportion'}, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 2_confusion_matrix.png")
    plt.close()
    
    return cm

def plot_per_class_metrics(y_true, y_pred, categories, output_dir):
    """Plot per-class precision, recall, and F1-score"""
    print("\n3. Creating per-class metrics plot...")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(categories))
    )
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Sign Language Letter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(output_dir, '3_per_class_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"   Saved: 3_per_class_metrics.png")
    except Exception as e:
        print(f"   Error saving 3_per_class_metrics.png: {e}")
    finally:
        plt.close()
    
    return precision, recall, f1, support

def plot_accuracy_comparison(y_true, y_pred, categories, output_dir):
    """Plot accuracy for each class"""
    print("\n4. Creating per-class accuracy plot...")
    
    accuracies = []
    for i, category in enumerate(categories):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            accuracies.append(acc * 100)
        else:
            accuracies.append(0)
    
    plt.figure(figsize=(16, 6))
    colors = ['green' if acc >= 90 else 'orange' if acc >= 70 else 'red' 
              for acc in accuracies]
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    plt.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    
    plt.xlabel('Sign Language Letter', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.ylim([0, 105])
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 4_per_class_accuracy.png")
    plt.close()

def plot_prediction_confidence(model, X_test, y_test, categories, output_dir):
    """Plot prediction confidence distribution"""
    print("\n5. Creating prediction confidence plot...")
    
    predictions = model.predict(X_test, verbose=0)
    max_confidences = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    correct_mask = predicted_classes == y_test
    correct_confidences = max_confidences[correct_mask]
    incorrect_confidences = max_confidences[~correct_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(correct_confidences, bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[0].hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data_to_plot = [correct_confidences, incorrect_confidences]
    axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Confidence Score Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_prediction_confidence.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 5_prediction_confidence.png")
    plt.close()

def plot_most_confused_pairs(cm, categories, output_dir, top_n=10):
    """Plot most confused letter pairs"""
    print("\n6. Creating most confused pairs plot...")
    
    confusion_pairs = []
    for i in range(len(categories)):
        for j in range(len(categories)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true': categories[i],
                    'predicted': categories[j],
                    'count': cm[i, j],
                    'pair': f"{categories[i]}→{categories[j]}"
                })
    
    confusion_pairs = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)[:top_n]
    
    if confusion_pairs:
        pairs = [item['pair'] for item in confusion_pairs]
        counts = [item['count'] for item in confusion_pairs]
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(pairs)))
        bars = plt.barh(pairs, counts, color=colors, edgecolor='black')
        plt.xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
        plt.ylabel('True → Predicted', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Confused Letter Pairs', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}',
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6_most_confused_pairs.png'), dpi=300, bbox_inches='tight')
        print(f"   Saved: 6_most_confused_pairs.png")
        plt.close()

def plot_sample_predictions(model, X_test, y_test, categories, output_dir, num_samples=16):
    """Plot sample predictions with images"""
    print("\n7. Creating sample predictions visualization...")
    
    # Get random samples
    indices = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)
    
    predictions = model.predict(X_test[indices], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, (i, ax) in enumerate(zip(indices, axes)):
        img = X_test[i].squeeze()
        true_label = categories[y_test[i]]
        pred_label = categories[predicted_classes[idx]]
        confidence = confidences[idx]
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}'
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_sample_predictions.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 7_sample_predictions.png")
    plt.close()

def plot_model_architecture_summary(model, output_dir):
    """Create a visual summary of model architecture"""
    print("\n8. Creating model architecture summary...")
    
    layer_info = []
    for layer in model.layers:
        try:
            output_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A'
        except:
            output_shape = 'N/A'
        
        layer_info.append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': output_shape,
            'params': layer.count_params()
        })
    
    df = pd.DataFrame(layer_info)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='left', loc='center',
                     colWidths=[0.25, 0.2, 0.3, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    total_params = model.count_params()
    plt.title(f'CNN Model Architecture Summary\nTotal Parameters: {total_params:,}', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_model_architecture.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 8_model_architecture.png")
    plt.close()

def generate_classification_report(y_true, y_pred, categories, output_dir):
    """Generate and save detailed classification report"""
    print("\n9. Generating classification report...")
    
    report = classification_report(y_true, y_pred, target_names=categories, digits=3)
    
    with open(os.path.join(output_dir, '9_classification_report.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"   Saved: 9_classification_report.txt")

def create_summary_dashboard(overall_acc, categories, output_dir):
    """Create a summary dashboard with key metrics"""
    print("\n10. Creating summary dashboard...")
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('CNN Model Performance Dashboard', fontsize=16, fontweight='bold')
    
    # Overall accuracy (large display)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.text(0.5, 0.5, f'{overall_acc*100:.2f}%', 
             ha='center', va='center', fontsize=60, fontweight='bold',
             color='green' if overall_acc >= 0.9 else 'orange' if overall_acc >= 0.7 else 'red')
    ax1.text(0.5, 0.15, 'Overall Accuracy', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Key statistics
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.text(0.5, 0.7, str(len(categories)), 
             ha='center', va='center', fontsize=36, fontweight='bold', color='blue')
    ax2.text(0.5, 0.3, 'Classes', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.text(0.5, 0.7, 'CNN', 
             ha='center', va='center', fontsize=36, fontweight='bold', color='purple')
    ax3.text(0.5, 0.3, 'Model Type', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.text(0.5, 0.7, f'{IMG_SIZE}x{IMG_SIZE}', 
             ha='center', va='center', fontsize=36, fontweight='bold', color='teal')
    ax4.text(0.5, 0.3, 'Input Size', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Info text
    ax5 = fig.add_subplot(gs[2, :])
    info_text = (
        "ASL Sign Language Recognition System\n"
        "Using Convolutional Neural Networks for A-Z Classification\n"
        f"Dataset: {len(categories)} letter categories | "
        f"Preprocessing: Grayscale → Blur → Adaptive Threshold → Otsu → Resize"
    )
    ax5.text(0.5, 0.5, info_text, 
             ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.axis('off')
    
    plt.savefig(os.path.join(output_dir, '10_summary_dashboard.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved: 10_summary_dashboard.png")
    plt.close()

def main():
    """Main function to generate all visualizations"""
    print("=" * 70)
    print("CNN MODEL PERFORMANCE VISUALIZATION SUITE")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_asl_model.py")
        return
    
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Load test data
    X_test, y_test, categories, label_dict, file_paths = load_test_data(DATASET_PATH)
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Generate all visualizations
    plot_dataset_distribution(y_test, categories, OUTPUT_DIR)
    cm = plot_confusion_matrix(y_test, y_pred, categories, OUTPUT_DIR)
    precision, recall, f1, support = plot_per_class_metrics(y_test, y_pred, categories, OUTPUT_DIR)
    plot_accuracy_comparison(y_test, y_pred, categories, OUTPUT_DIR)
    plot_prediction_confidence(model, X_test, y_test, categories, OUTPUT_DIR)
    plot_most_confused_pairs(cm, categories, OUTPUT_DIR)
    plot_sample_predictions(model, X_test, y_test, categories, OUTPUT_DIR)
    plot_model_architecture_summary(model, OUTPUT_DIR)
    generate_classification_report(y_test, y_pred, categories, OUTPUT_DIR)
    create_summary_dashboard(overall_accuracy, categories, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. 1_dataset_distribution.png - Dataset class distribution")
    print("  2. 2_confusion_matrix.png - Confusion matrices (raw & normalized)")
    print("  3. 3_per_class_metrics.png - Precision, Recall, F1-Score per class")
    print("  4. 4_per_class_accuracy.png - Accuracy breakdown by letter")
    print("  5. 5_prediction_confidence.png - Confidence score analysis")
    print("  6. 6_most_confused_pairs.png - Most commonly confused letters")
    print("  7. 7_sample_predictions.png - Visual prediction samples")
    print("  8. 8_model_architecture.png - CNN architecture summary")
    print("  9. 9_classification_report.txt - Detailed metrics report")
    print("  10. 10_summary_dashboard.png - Overall performance dashboard")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
