# DQL_multirotor_landing

Implementation of Double Q-Learning with Curriculum Learning for autonomous UAV landing on a moving platform. The approach decomposes the landing task into sequential sub-tasks and applies state-space discretization to improve learning efficiency and maneuverability.

## Settingup with uv and VSCode on Linux

```bash
# Creates a new venv and install the needed dependencies
uv sync
# Activate the said .venv
source .venv/bin/activate
```

<!-- Git init submodule recursive -->

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
./IsaacLab/isaaclab.sh --install
# Solves the issues with the extrapaths
cp -r ./IsaacLab/.vscode ./.venv/lib/python3.10/site-packages/isaacsim/.vsco
de/
```

## Setting Up IsaacLab with Pyenv and VSCode

If you are using **Pyenv** to manage the Python virtual environment, follow these additional steps to ensure that **VSCode** correctly recognizes the paths for **Isaac Sim** and **IsaacLab**.

### **1. Activate the Virtual Environment**

Make sure to activate the virtual environment before proceeding:

```bash
pyenv activate <your_env_name>
```

### **2. Install Isaac Sim and IsaacLab**

Follow the official installation guide for **IsaacLab**:
👉 [IsaacLab Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

### **3. Create the `.vscode` Folder in Isaac Sim**

Navigate to the **Isaac Sim** package directory within the virtual environment:

```bash
cd ~/.pyenv/versions/<python_version(3.10.15)>/envs/<your_env_name>/lib/python<python_version(python3.10.15)>/site-packages/isaacsim/
```

Create the `.vscode` folder and the `settings.json` file:

```bash
mkdir -p .vscode
touch .vscode/settings.json
```

### **4. Configure VSCode for IsaacLab**

- Open **VSCode** and load the **IsaacLab** folder.
- Press `Shift + Ctrl + P` to open the **Command Palette**.
- Search for and execute:

  ```
  Run Tasks → setup_python_env.sh
  ```

- Run the setup script:

  ```bash
  ./IsaacLab/.vscode/tools/setup_vscode.py
  ```

### **5. Verify Path Recognition in VSCode**

At this point, **VSCode** should correctly recognize the paths. To verify:

- Open a Python file in VSCode and check if the **Omniverse/Isaac Sim** libraries are recognized without errors.
- If the code runs correctly but **intellisense** does not work:
  1. Create a `.vscode` folder in the root of your workspace:

     ```bash
     mkdir -p <your_workspace>/.vscode
     ```

  2. Copy the contents of `~/.pyenv/versions/<python_version>/envs/<your_env_name>/lib/python<python_version>/site-packages/isaacsim/.vscode/settings.json` into a new `settings.json` file inside your workspace:

     ```bash
     cp ~/.pyenv/versions/<python_version>/envs/<your_env_name>/lib/python<python_version>/site-packages/isaacsim/.vscode/settings.json <your_workspace>/.vscode/settings.json
     ```

After completing these steps, **VSCode will correctly recognize all Isaac Sim and IsaacLab modules**
