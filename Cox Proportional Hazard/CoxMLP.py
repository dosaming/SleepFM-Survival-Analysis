
from __future__ import annotations
import argparse, pickle, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines.utils import concordance_index


def load_survival_pickle(pickle_path: str):
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)

    key_x = "x"
    key_t = "t" if "t" in d else ("duration_days" if "duration_days" in d else None)
    key_e = "e" if "e" in d else ("event" if "event" in d else None)
    if key_t is None or key_e is None or key_x not in d:
        raise ValueError(
            f"키 없는 에러: {list(d.keys())}, need: x, t(or duration_days), e(or event)"
        )

    x = np.asarray(d[key_x], dtype=np.float32)
    t = np.asarray(d[key_t], dtype=np.float32)
    e = np.asarray(d[key_e], dtype=np.int32)

    mask = np.isfinite(x).all(axis=1) & np.isfinite(t) & np.isfinite(e)
    return {"x": x[mask], "t": t[mask], "e": e[mask]}


class CoxMLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout=0.0, batch_norm=False):
        super().__init__()

        layers = []
        last = n_in
        for h in hidden:
            layers.append(nn.Linear(last, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())  # 항상 ReLU 사용
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))  # log-risk 출력층
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (N,)



def neg_partial_log_likelihood(risk_scores: torch.Tensor,
                               t: torch.Tensor,
                               e: torch.Tensor,
                               ties: str = "efron") -> torch.Tensor:

    # 내림차순 정렬을 해서 리스트셋을 앞부분으로 옮김
    order = torch.argsort(t, descending=True)
    r = risk_scores[order]
    ee = e[order].float()
    tt = t[order]

    #지수 계산하고 누적합 계산
    exp_r = torch.exp(r)
    cum_sum = torch.cumsum(exp_r, dim=0)

    #사건 발생한 1 샘플만 뽑음
    event_idx = torch.nonzero(ee > 0.5, as_tuple=False).squeeze(-1)
    if event_idx.numel() == 0:
        return torch.tensor(0.0, dtype=r.dtype, device=r.device, requires_grad=True)

    unique_times, inverse = torch.unique(tt[event_idx], return_inverse=True)
    losses = []
    # tie event 처리
    for k in range(len(unique_times)):
        mask = (inverse == k)
        idx_k = event_idx[mask]           
        d = idx_k.numel()                 
        sum_r_events = torch.sum(r[idx_k])

        j = torch.max(idx_k)
        denom = cum_sum[j]

        #breslow: 처음 나온 거. 동시 있어도 한번에 처리
        if ties.lower() == "breslow":
            loss_k = sum_r_events - d * torch.log(torch.clamp(denom, min=1e-12))
        #efron: 순차적으로 일어났다고 가정하고 계산 
        elif ties.lower() == "efron":
            tied_exp_sum = torch.sum(exp_r[idx_k])
            l_range = torch.arange(d, dtype=r.dtype, device=r.device)
            denom_terms = denom - (l_range / d) * tied_exp_sum
            denom_terms = torch.clamp(denom_terms, min=1e-12)
            loss_k = sum_r_events - torch.sum(torch.log(denom_terms))
        else:
            raise ValueError("잘못입력")

        losses.append(loss_k)
    #사건 개수로 나눠서 손실 리턴
    total = torch.sum(torch.stack(losses))
    num_events = torch.sum(ee)
    return -(total / (num_events + 1e-8))  # negative mean log-likelihood


def standardize_fit(x: np.ndarray):
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std

# train 기준으로 정규화 하는 게 맞나 .. 
def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (x - mean) / std

def c_index_torch(model: nn.Module, x: np.ndarray, t: np.ndarray, e: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(x).to(device)
        scores = model(X).detach().cpu().numpy()
 
    return float(concordance_index(t, -scores, e))


def train(run_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train = load_survival_pickle(run_args.train_pickle)
    valid = load_survival_pickle(run_args.valid_pickle)
    test = load_survival_pickle(run_args.test_pickle) if run_args.test_pickle else None


    mean, std = standardize_fit(train["x"])
    x_tr = standardize_apply(train["x"], mean, std)
    x_va = standardize_apply(valid["x"], mean, std)
    if test is not None:
        x_te = standardize_apply(test["x"], mean, std)

    
    X_tr = torch.from_numpy(x_tr).to(device)
    T_tr = torch.from_numpy(train["t"]).to(device)
    E_tr = torch.from_numpy(train["e"].astype(np.float32)).to(device)

    X_va = torch.from_numpy(x_va).to(device)
    T_va = torch.from_numpy(valid["t"]).to(device)
    E_va = torch.from_numpy(valid["e"].astype(np.float32)).to(device)


    hidden = [int(s) for s in run_args.hidden.split(",") if s.strip()] if run_args.hidden else []
    model = CoxMLP(
        n_in=X_tr.shape[1],
        hidden=hidden,
        dropout=run_args.dropout,
        batch_norm=run_args.batch_norm,
    ).to(device)

   
    opt = optim.Adam(model.parameters(), lr=run_args.lr, weight_decay=run_args.l2)  
    use_l1 = run_args.l1 > 0.0

    best_val_loss = math.inf
    best_val_c = None
    best_state = None
    patience = run_args.patience
    start = time.time()

    for epoch in range(1, run_args.epochs + 1):
        model.train()
        opt.zero_grad()
        scores = model(X_tr)  
        loss = neg_partial_log_likelihood(scores, T_tr, E_tr, ties=run_args.ties)
        if use_l1:
            l1_pen = sum(p.abs().sum() for p in model.parameters())
            loss = loss + run_args.l1 * l1_pen
        loss.backward()
        if run_args.grad_clip is not None and run_args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), run_args.grad_clip)
        opt.step()

        if epoch % run_args.val_freq == 0:
            model.eval()
            with torch.no_grad():
                tr_c = c_index_torch(model, x_tr, train["t"], train["e"], device)
                val_scores = model(X_va)
                val_loss = neg_partial_log_likelihood(val_scores, T_va, E_va, ties=run_args.ties).item()
                va_c = c_index_torch(model, x_va, valid["t"], valid["e"], device)
            print(f"[{epoch:04d}] loss {loss.item():.6f} | train c {tr_c:.4f} | valid loss {val_loss:.6f} | valid c {va_c:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_c = va_c
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = max(patience, epoch * run_args.patience_increase)

        if epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    tr_c = c_index_torch(model, x_tr, train["t"], train["e"], device)
    va_c = c_index_torch(model, x_va, valid["t"], valid["e"], device)

    print("\n=== C-Index ===")
    print(f"train: {tr_c:.4f}")
    print(f"valid: {va_c:.4f}")
    if best_val_c is not None:
        print(f"best(valid): {best_val_c:.4f}")

    if test is not None:
        te_c = c_index_torch(model, x_te, test["t"], test["e"], device)
        print(f"test : {te_c:.4f}")


    if run_args.out_path:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "config": vars(run_args),
            },
            run_args.out_path,
        )
        print(f"[SAVE] {run_args.out_path}")



def build_parser():
    p = argparse.ArgumentParser("Cox Deep (PyTorch) — pre-split pickles (train/valid[/test])")
    p.add_argument("--train_pickle", type=str, required=True)
    p.add_argument("--valid_pickle", type=str, required=True)
    p.add_argument("--test_pickle", type=str, default=None)


    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-4, help="L2 weight decay")
    p.add_argument("--l1", type=float, default=0.0)
    p.add_argument("--hidden", type=str, default="256,128")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_norm", action="store_true")

  
    p.add_argument("--val_freq", type=int, default=10)
    p.add_argument("--patience", type=int, default=2000)
    p.add_argument("--patience_increase", type=int, default=2)
    p.add_argument("--ties", type=str, default="efron", choices=["efron","breslow"])

    
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--out_path", type=str, default=None)
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
