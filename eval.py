#############################
#      MODEL EVALUATION     #
#############################

@torch.no_grad()
def evaluate(model, dataloader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    latencies = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        start = time.time()
        out = model(x)
        latencies.append(time.time() - start)

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / total
    avg_latency = sum(latencies) / len(latencies)
    return acc, avg_latency