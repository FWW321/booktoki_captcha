跳过burn的libtorch版本检测
```
$env:LIBTORCH_BYPASS_VERSION_CHECK = 1
```

训练
```
cargo run --release --bin train --features training
```

推理
```
cargo run --release --bin booktoki_captcha
```