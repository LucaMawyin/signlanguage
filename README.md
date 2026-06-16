<project>
    <warning>
        <title>CAUTION</title>
        <description>
            The YOLO folder contains 5600 images of training data and 5600 images of parsed images.
            User discretion is advised when opening.
        </description>
    </warning>

    <dependencies>
        <install>
            <command>pip install ultralytics</command>
        </install>

        <install>
            <command>pip install mediapipe==0.10.14</command>
        </install>

        <install>
            <command>pip install opencv-python</command>
        </install>

        <install>
            <command>pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121</command>
            <note>PyTorch with CUDA 12.1 support</note>
        </install>
    </dependencies>

    <execution_order>
        <step number="1">
            <file>create_labels.py</file>
            <description>
                Tracks hand skeletons and creates YOLO label TXT files.
            </description>
        </step>

        <step number="2">
            <file>yolo_model.py</file>
            <description>
                Trains the YOLO model for 50 epochs.
            </description>
                <note>
                    Ensure GPU is enabled (CUDA recommended) & modify settings for user device specs
                </note>
        </step>
    </execution_order>

</project>
