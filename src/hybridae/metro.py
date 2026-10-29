"""Metro Interstate Traffic Volume pipeline shared by HAE/run_metro.py and VHAE/run_metro.py."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from .data import load_hourly_traffic, build_metro_daily, weather_target_columns
from .forecast import forecast_scores_time_series
from .metrics import mse


def prepare_metro(path, horizon: int = 7):
    s, x_counts, y_raw, names, dates = build_metro_daily(load_hourly_traffic(path))
    L = horizon
    x_log = np.log1p(x_counts).astype(np.float32)
    xs = StandardScaler(); ys = StandardScaler()          # fitted on pre-forecast days only
    x_tr = xs.fit_transform(x_log[:-L]).astype(np.float32); x_te = xs.transform(x_log[-L:]).astype(np.float32)
    y_tr = ys.fit_transform(y_raw[:-L]).astype(np.float32); y_te = ys.transform(y_raw[-L:]).astype(np.float32)
    return dict(s=s, x_counts=x_counts, y_raw=y_raw, names=names, dates=dates, L=L,
                x_tr=x_tr, x_te=x_te, y_tr=y_tr, y_te=y_te, x_scaler=xs, y_scaler=ys,
                x_te_counts=x_counts[-L:], weather_idx=weather_target_columns(names))


def to_counts(x_std, xs):
    return np.clip(np.expm1(xs.inverse_transform(np.asarray(x_std, dtype=np.float64))), 0.0, None)


@torch.no_grad()
def hae_recon_and_forecast(model, D, cfg):
    """Encode the pre-forecast block (posterior means for VHAE), fit AIC-VAR to the
    latent scores, decode the latent forecasts."""
    xt = torch.tensor(D["x_tr"]); yt = torch.tensor(D["y_tr"])
    model.eval()
    x_hat, y_hat, H, _ = model(xt, yt)
    Hf, info, scaler, res = forecast_scores_time_series(H.numpy(), D["L"], cfg.max_ar_lags, return_model=True)
    xf, _, yf = model.decode_latent(torch.tensor(Hf, dtype=torch.float32))
    return dict(x_recon=x_hat.numpy(), y_recon=y_hat.numpy(), H=H.numpy(), H_fore=Hf, info=info,
                x_fore=xf.numpy(), y_fore=yf.numpy(), scaler=scaler, var_result=res)


def nonfunctional_blocks(names):
    """Variable blocks of Table 'metro_variables' (paper): weather (29), calendar (9), holiday (11).
    Weather is split further into continuous summaries (18) and indicators (11)."""
    w = weather_target_columns(names)
    cont = [i for i in w if not (names[i].startswith("weather_") or names[i] in ("rain_any", "snow_any"))]
    ind = [i for i in w if i not in cont]
    hol = [i for i, n in enumerate(names) if n.startswith("holiday_")]
    cal = [i for i in range(len(names)) if i not in w and i not in hol]
    return {"weather": w, "weather_continuous": cont, "weather_indicator": ind,
            "calendar": cal, "holiday": hol}


def per_variable_mse(pred, truth):
    return np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) ** 2, axis=0)


def point_table(D, x_recon, y_recon, x_fore, y_fore):
    """MSEs on the standardized scale (Table 'traffic_hybrid').

    recon_weather / fore_weather : mean over the 29 weather variables only.
    contrib_<block>_{recon,fore}  : contribution of a block to the nonfunctional MSE,
        i.e. sum_{j in block} MSE_j / p, as reported in the paper for the weather block
        (the contributions of the three blocks add up to recon_nonfunc / fore_nonfunc).
    """
    w = D["weather_idx"]
    out = dict(
        recon_func=mse(x_recon, D["x_tr"]), recon_nonfunc=mse(y_recon, D["y_tr"]),
        recon_hybrid=mse(x_recon, D["x_tr"]) + mse(y_recon, D["y_tr"]),
        fore_func=mse(x_fore, D["x_te"]), fore_nonfunc=mse(y_fore, D["y_te"]),
        fore_hybrid=mse(x_fore, D["x_te"]) + mse(y_fore, D["y_te"]),
        fore_func_counts=mse(to_counts(x_fore, D["x_scaler"]), D["x_te_counts"]),
        recon_weather=float(np.mean((np.asarray(y_recon)[:, w] - D["y_tr"][:, w]) ** 2)),
        fore_weather=float(np.mean((np.asarray(y_fore)[:, w] - D["y_te"][:, w]) ** 2)),
    )
    p = D["y_tr"].shape[1]
    vr, vf = per_variable_mse(y_recon, D["y_tr"]), per_variable_mse(y_fore, D["y_te"])
    for b, idx in nonfunctional_blocks(D["names"]).items():
        out[f"contrib_{b}_recon"] = float(vr[idx].sum() / p)
        out[f"contrib_{b}_fore"] = float(vf[idx].sum() / p)
    return out
