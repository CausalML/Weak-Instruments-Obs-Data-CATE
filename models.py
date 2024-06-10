import torch
import torch.nn as nn

# Validation utility
class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_dict = None  # Holds the best model state dictionary

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.best_model_dict = model.state_dict()  # Save the model state
        self.val_loss_min = val_loss

# Representation network
class PhiNet(nn.Module):
    def __init__(self, n_cov, n_hidden, d_hidden, d_out):
        super(PhiNet, self).__init__()
        layers = []
        for k in range(n_hidden):
            in_features = n_cov if k == 0 else d_hidden
            out_features = d_out if k == (n_hidden - 1) else d_hidden
            layers.append(nn.Linear(in_features, out_features))
            if k < n_hidden-1:
                layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class CATENet(nn.Module):
    def __init__(self, n_cov, n_hidden, d_hidden, d_out):
        super(CATENet, self).__init__()
        # Define representation network
        self.rep = PhiNet(n_cov, n_hidden, d_hidden, d_out)

        # Linear heads for treatment and control outcomes
        self.linear_out_treatment = nn.Linear(d_out, 1)
        self.linear_out_control = nn.Linear(d_out, 1)

    def forward(self, x):
        # Generate shared representation from input features
        shared_rep = self.rep(x)

        # Generate outcomes for treatment and control
        treatment_output = self.linear_out_treatment(shared_rep).squeeze(-1)
        control_output = self.linear_out_control(shared_rep).squeeze(-1)

        return treatment_output, control_output

    def train_model(self, x, a, y, x_val, a_val, y_val, learning_rate, n_epochs, lambda_y=0.01, batch_size=64, eval_interval=5, patience=5, verbose=True):
        optimizer = torch.optim.Adam([
            {'params': self.rep.parameters(), 'weight_decay': 0.02},
            {'params': self.linear_out_treatment.parameters(), 'weight_decay': 0.00},
            {'params': self.linear_out_control.parameters(), 'weight_decay': 0.00}
            ], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        loss_function = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience, verbose=verbose)

        best_val_loss = float('inf')
        best_model = None

        dataset_size = x.shape[0]
        indices = torch.randperm(dataset_size)  # Shuffle indices

        for epoch in range(n_epochs):
            self.train()  # Set model to training mode
            total_loss = 0

            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_x = x[batch_indices]
                batch_a = a[batch_indices]
                batch_y = y[batch_indices]

                optimizer.zero_grad()

                treatment_pred, control_pred = self.forward(batch_x)

                treatment_mask = (batch_a == 1)
                control_mask = (batch_a == 0)

                treatment_loss = loss_function(treatment_pred[treatment_mask], batch_y[treatment_mask])
                control_loss = loss_function(control_pred[control_mask], batch_y[control_mask])

                loss = treatment_loss + control_loss

                #Regularize the linear hypothesis
                reg1 = torch.norm(self.linear_out_treatment.weight, p=2) + torch.norm(self.linear_out_treatment.bias, p=2)
                reg0 = torch.norm(self.linear_out_control.weight, p=2) + torch.norm(self.linear_out_control.bias, p=2)
                reg_w = reg1 + reg0

                loss_total = loss + lambda_y * reg_w
                loss_total.backward()
                optimizer.step()

                total_loss += loss_total.item() * (end_idx - start_idx)  # Weight loss by batch size

            average_loss = total_loss / dataset_size  # Calculate average loss over the epoch

            if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                print(f'Epoch {epoch+1}, Loss: {average_loss}')

            # Validation and early stopping logic
            if epoch % eval_interval == 0:
                self.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    # Predict validation outcomes
                    treatment_pred_val, control_pred_val = self.forward(x_val)
                    # Calculate validation losses
                    treatment_loss_val = loss_function(treatment_pred_val[a_val == 1], y_val[a_val == 1])
                    control_loss_val = loss_function(control_pred_val[a_val == 0], y_val[a_val == 0])
                    val_loss = treatment_loss_val + control_loss_val

                    # Check if the current validation loss is the best one
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = self.state_dict()  # Save the best model

                if verbose:
                  # Print loss for monitoring
                  print(f'Epoch {epoch+1}, Training Loss: {average_loss}, Validation Loss: {val_loss}')

                early_stopping(val_loss, self)
                if early_stopping.early_stop:
                    if early_stopping.best_model_dict:
                        self.load_state_dict(early_stopping.best_model_dict)
                    print("Early stopping")
                    break


class ComplianceNet(nn.Module):
    def __init__(self, n_cov, n_hidden, d_hidden, d_out):
        super(ComplianceNet, self).__init__()
        # Define representation network
        self.rep = PhiNet(n_cov, n_hidden, d_hidden, d_out)

        # Linear head for propensity score estimation
        self.linear_treatment_z_1 = nn.Linear(d_out, 1)
        self.linear_treatment_z_0 = nn.Linear(d_out, 1)

    def forward(self, x):
        # Generate shared representation from input features
        shared_rep = self.rep(x)

        # Generate outcomes for treatment and control
        treatment_z_1 = torch.sigmoid(self.linear_treatment_z_1(shared_rep).squeeze(-1))
        treatment_z_0 = torch.sigmoid(self.linear_treatment_z_0(shared_rep).squeeze(-1))

        return treatment_z_1, treatment_z_0

    def train_model(self, x, z, a, x_val, z_val, a_val, learning_rate, n_epochs, lambda_y=0.01, batch_size=64, eval_interval=5, patience=5, verbose=True):
        optimizer = torch.optim.Adam([
            {'params': self.rep.parameters(), 'weight_decay': 0.01}
            ], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        bce_loss = nn.BCELoss()
        early_stopping = EarlyStopping(patience=patience, verbose=verbose)

        best_val_loss = float('inf')
        best_model = None

        dataset_size = x.shape[0]
        indices = torch.randperm(dataset_size)  # Shuffle indices

        for epoch in range(n_epochs):
            self.train()  # Set model to training mode
            total_loss = 0

            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_x = x[batch_indices]
                batch_z = z[batch_indices]
                batch_a = a[batch_indices]

                optimizer.zero_grad()

                treatment_z_1, treatment_z_0 = self.forward(batch_x)

                z_1_mask = (batch_z == 1)
                z_0_mask = (batch_z == 0)
                t_z_1_loss = bce_loss(treatment_z_1[z_1_mask], batch_a[z_1_mask])
                t_z_0_loss = bce_loss(treatment_z_0[z_0_mask], batch_a[z_0_mask])

                loss = t_z_1_loss + t_z_0_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * (end_idx - start_idx)  # Weight loss by batch size

            average_loss = total_loss / dataset_size  # Calculate average loss over the epoch

            if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                print(f'Epoch {epoch+1}, Loss: {average_loss}')

            # Validation and early stopping logic
            if epoch % eval_interval == 0:
                self.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    # Predict validation outcomes
                    treatment_z_1_val, treatment_z_0_val = self.forward(x_val)
                    # Calculate validation losses
                    t_z_1_loss_val = bce_loss(treatment_z_1_val[z_val==1], a_val[z_val==1])
                    t_z_0_loss_val = bce_loss(treatment_z_0_val[z_val==0], a_val[z_val==0])

                    val_loss = t_z_1_loss_val + t_z_0_loss_val
                    # Check if the current validation loss is the best one
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = self.state_dict()  # Save the best model

                if verbose:
                  # Print loss for monitoring
                  print(f'Epoch {epoch+1}, Training Loss: {average_loss}, Validation Loss: {val_loss}')

                early_stopping(val_loss, self)
                if early_stopping.early_stop:
                    if early_stopping.best_model_dict:
                        self.load_state_dict(early_stopping.best_model_dict)
                    print("Early stopping")
                    break


class ComplianceNet_OneStep(nn.Module):
    def __init__(self, n_cov, n_hidden, d_hidden, d_out):
        super(ComplianceNet_OneStep, self).__init__()
        # Define representation network
        self.rep = PhiNet(n_cov, n_hidden, d_hidden, d_out)

        # Linear head for propensity score estimation
        self.linear_gamma = nn.Linear(d_out, 1)

    def forward(self, x):
        # Generate shared representation from input features
        shared_rep = self.rep(x)

        # Generate outcomes for treatment and control
        gamma = torch.sigmoid(self.linear_gamma(shared_rep).squeeze(-1))

        return gamma

    def train_model(self, x, c, x_val, c_val, learning_rate, n_epochs, lambda_y=0.01, batch_size=64, eval_interval=5, patience=5, verbose=True):
        optimizer = torch.optim.Adam([
            {'params': self.rep.parameters(), 'weight_decay': 0.00}
            ], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        mse_loss = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience, verbose=verbose)

        best_val_loss = float('inf')
        best_model = None

        dataset_size = x.shape[0]
        indices = torch.randperm(dataset_size)  # Shuffle indices

        for epoch in range(n_epochs):
            self.train()  # Set model to training mode
            total_loss = 0

            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_x = x[batch_indices]
                batch_c = c[batch_indices]

                optimizer.zero_grad()

                gamma = self.forward(batch_x)
                loss = mse_loss(gamma, batch_c)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * (end_idx - start_idx)  # Weight loss by batch size

            average_loss = total_loss / dataset_size  # Calculate average loss over the epoch

            if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                print(f'Epoch {epoch+1}, Loss: {average_loss}')

            # Validation and early stopping logic
            if epoch % eval_interval == 0:
                self.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    # Predict validation outcomes
                    gamma_val = self.forward(x_val)
                    # Calculate validation losses
                    val_loss = mse_loss(gamma_val, c_val)
                    # Check if the current validation loss is the best one
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = self.state_dict()  # Save the best model

                if verbose:
                  # Print loss for monitoring
                  print(f'Epoch {epoch+1}, Training Loss: {average_loss}, Validation Loss: {val_loss}')

                early_stopping(val_loss, self)
                if early_stopping.early_stop:
                    if early_stopping.best_model_dict:
                      self.load_state_dict(early_stopping.best_model_dict)
                    print("Early stopping")
                    break