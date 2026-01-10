# CLMDA：Cross Language Vulnerability Detection Based on Multimodal Learning and Domain Adaptation

## 📖 Project Overview

This project is a comprehensive multi-modal code analysis framework designed to process and analyze source code through multiple perspectives including text sequences, dynamic paths, and graph structures. The system integrates multiple machine learning techniques for cross-domain code analysis and vulnerability detection.

## 🏗️ Project Architecture
`1
1
1`

├── Text_Sequence/          # Text sequence processing module
│   ├── or_ast_generator.py            # Original code AST generation
│   ├── or_ast_to_sequence.py          # AST to sequence conversion
│   ├── or_ast_sequence_labeling.py    # AST sequence labeling
│   ├── or_data_processor.py           # Original data processing
│   ├── or_encoder_trainer.py          # Original encoder training
│   ├── ta_ast_generator.py            # Target code AST generation
│   ├── ta_ast_to_sequence.py          # Target AST to sequence
│   ├── ta_ast_sequence_labeling.py    # Target AST sequence labeling
│   ├── ta_data_processor.py           # Target data processing
│   ├── ta_encoder_trainer.py          # Target encoder training
│   ├── data_augmentation.py           # Data augmentation
│   ├── dann_encoder_trainer.py        # DANN encoder training
│   └── run_all_scripts.bat            # One-click execution script
│
├── Dynamic_Path/          # Dynamic execution path analysis
│   ├── or_get_dy_label.py             # Original dynamic label extraction
│   ├── or_data_process.py             # Original data processing
│   ├── or_transformer_encoder_dy.py   # Original transformer encoder
│   ├── ta_get_dy_label.py             # Target dynamic label extraction
│   ├── ta_data_process.py             # Target data processing
│   ├── ta_transformer_encoder_dy.py   # Target transformer encoder
│   ├── data_aug.py                    # Data augmentation
│   ├── encoder_dann_train.py          # DANN encoder training
│   └── run_all_scripts.bat            # One-click execution script
│
├── Figure_Structure/      # Graph structure analysis
│   ├── or_generate_CPG.py             # Original CPG generation
│   ├── or_make_to_graphData.py        # Original graph data creation
│   ├── or_encoder_GAT_cpg.py          # Original GAT encoder
│   ├── ta_generate_CPG.py             # Target CPG generation
│   ├── ta_make_to_graphData.py        # Target graph data creation
│   ├── ta_encoder_GAT_cpg.py          # Target GAT encoder
│   ├── data_aug.py                    # Data augmentation
│   ├── encode_dann_train.py           # DANN encoder training
│   └── run_all_scripts.bat            # One-click execution script
│
├── Feature_Fusion/        # Multi-modal feature fusion
│   ├── fuse_e3_S_D_C.py               # Feature fusion of S, D, C modalities
│   ├── Classification_e3_S_D_C.py     # Final classification
│   └── run_all_scripts.bat            # One-click execution script
│
└── run_all_scripts.bat    # Main controller script (run all modules)

## 🚀 Getting Started

### Prerequisites
Python 3.8+
Required Python packages (see requirements.txt)

### Installation
Clone the repository:
`
git clone https://github.com/yourusername/code-analysis-framework.git
cd code-analysis-framework
`

Install required dependencies:
`
pip install -r requirements.txt
`

Ensure all required data files are in place (consult data/README.md)



