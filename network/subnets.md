# Task 5: Subnetting and Network Access Control

## 1. Network Overview and Addressing Requirements

Base CIDR block allocated: **`10.20.0.0/16`**  
Total available address space: $2^{32-16} = 2^{16} = 65,536$ IP addresses.

Per project specification, subnets are allocated sequentially from largest host requirement to smallest host requirement, beginning strictly at `10.20.0.0`, with each subnet placed immediately after the previous one using the smallest CIDR prefix that satisfies the host requirement. Usable hosts formula:
$$\text{Usable Hosts} = 2^{32 - \text{prefix}} - 2$$

---

## 2. Step-by-Step Subnet Allocation & Working

### Subnet 1: `app` (Largest requirement)
- **Hosts Needed:** 1,000
- **Prefix Calculation:**  
  Find smallest host bit count $h$ such that $2^h - 2 \ge 1,000$:  
  $2^9 - 2 = 510$ (insufficient)  
  $2^{10} - 2 = 1,022 \ge 1,000$ (satisfies requirement)  
  Host bits $h = 10 \implies$ Prefix length = $32 - 10 = \mathbf{/22}$.
- **Network Block:** Starting at `10.20.0.0`
- **Subnet Mask:** `255.255.252.0`
- **Address Block Size:** $2^{10} = 1,024$ addresses (`10.20.0.0` to `10.20.3.255`)
- **Usable Host Range:** `10.20.0.1` – `10.20.3.254`
- **Broadcast Address:** `10.20.3.255`
- **Number of Usable Hosts:** $1,024 - 2 = \mathbf{1,022}$

---

### Subnet 2: `public` (Second largest requirement)
- **Hosts Needed:** 250
- **Placement:** Begins immediately after `app` subnet broadcast at `10.20.4.0`.
- **Prefix Calculation:**  
  Find smallest host bit count $h$ such that $2^h - 2 \ge 250$:  
  $2^7 - 2 = 126$ (insufficient)  
  $2^8 - 2 = 254 \ge 250$ (satisfies requirement)  
  Host bits $h = 8 \implies$ Prefix length = $32 - 8 = \mathbf{/24}$.
- **Network Block:** `10.20.4.0/24`
- **Subnet Mask:** `255.255.255.0`
- **Address Block Size:** $2^8 = 256$ addresses (`10.20.4.0` to `10.20.4.255`)
- **Usable Host Range:** `10.20.4.1` – `10.20.4.254`
- **Broadcast Address:** `10.20.4.255`
- **Number of Usable Hosts:** $256 - 2 = \mathbf{254}$

---

### Subnet 3: `data` (Third largest requirement)
- **Hosts Needed:** 100
- **Placement:** Begins immediately after `public` subnet broadcast at `10.20.5.0`.
- **Prefix Calculation:**  
  Find smallest host bit count $h$ such that $2^h - 2 \ge 100$:  
  $2^6 - 2 = 62$ (insufficient)  
  $2^7 - 2 = 126 \ge 100$ (satisfies requirement)  
  Host bits $h = 7 \implies$ Prefix length = $32 - 7 = \mathbf{/25}$.
- **Network Block:** `10.20.5.0/25`
- **Subnet Mask:** `255.255.255.128`
- **Address Block Size:** $2^7 = 128$ addresses (`10.20.5.0` to `10.20.5.127`)
- **Usable Host Range:** `10.20.5.1` – `10.20.5.126`
- **Broadcast Address:** `10.20.5.127`
- **Number of Usable Hosts:** $128 - 2 = \mathbf{126}$

---

### Subnet 4: `management` (Smallest requirement)
- **Hosts Needed:** 20
- **Placement:** Begins immediately after `data` subnet broadcast at `10.20.5.128`.
- **Prefix Calculation:**  
  Find smallest host bit count $h$ such that $2^h - 2 \ge 20$:  
  $2^4 - 2 = 14$ (insufficient)  
  $2^5 - 2 = 30 \ge 20$ (satisfies requirement)  
  Host bits $h = 5 \implies$ Prefix length = $32 - 5 = \mathbf{/27}$.
- **Network Block:** `10.20.5.128/27`
- **Subnet Mask:** `255.255.255.224`
- **Address Block Size:** $2^5 = 32$ addresses (`10.20.5.128` to `10.20.5.159`)
- **Usable Host Range:** `10.20.5.129` – `10.20.5.158`
- **Broadcast Address:** `10.20.5.159`
- **Number of Usable Hosts:** $32 - 2 = \mathbf{30}$

---

## 3. Subnet Allocation Table

| Subnet | Network Address | Prefix Length | Usable Host Range | Broadcast Address | Usable Hosts ($2^{32-\text{prefix}}-2$) |
|:---|:---|:---:|:---|:---|:---:|
| **app** | `10.20.0.0` | `/22` | `10.20.0.1` – `10.20.3.254` | `10.20.3.255` | 1,022 |
| **public** | `10.20.4.0` | `/24` | `10.20.4.1` – `10.20.4.254` | `10.20.4.255` | 254 |
| **data** | `10.20.5.0` | `/25` | `10.20.5.1` – `10.20.5.126` | `10.20.5.127` | 126 |
| **management** | `10.20.5.128` | `/27` | `10.20.5.129` – `10.20.5.158` | `10.20.5.159` | 30 |

All subnets are contiguous, non-overlapping, and fully contained within `10.20.0.0/16`.

---

## 4. Architectural Reasoning: Stateless NACL & Rule Ordering

### Why a Stateless NACL Requires Explicit Rules for Reply Traffic
Unlike stateful inspection firewalls (or Kubernetes Security Groups) that maintain an internal connection state tracking table (tracking TCP sequence numbers, SYN/ACK handshakes, and ephemeral port tuples), a Network Access Control List (NACL) is completely **stateless**. 

In a stateless firewall architecture:
1. Each incoming and outgoing network packet is evaluated in isolation against the defined rule table, without any memory of previous packets or established sessions.
2. When an external client initiates an inbound TCP connection to the public load balancer on port 80 or 443, the inbound packet is permitted by the inbound allow rules. However, the load balancer's return packet travels back towards the client's temporary ephemeral port (typically in the range `1024`–`65535`).
3. Because the NACL has no state table recording that this outbound packet is part of an existing, legitimately initiated TCP conversation, the outbound packet is matched afresh against outbound rules. Without an explicit outbound rule permitting traffic to ephemeral ports `1024`–`65535`, the packet would be dropped by the default outbound deny rule, resulting in a broken connection timeout for the user.
4. Likewise, when the load balancer forwards requests internally to the backend `app` subnet on TCP port 8000 (outbound rule 100), the app tier's return responses arrive back at the public subnet on ephemeral ports (`1024`–`65535`). An explicit inbound rule (rule 130) is required to admit those return packets.

### Why Rule Order Matters for the `198.51.100.0/24` Block
NACL evaluation follows a strict **first-match-wins** algorithm:
1. Rules are evaluated sequentially in ascending numerical order of their rule numbers.
2. As soon as a packet matches the criteria (protocol, port range, and CIDR) of a rule, that rule's action (`allow` or `deny`) is immediately executed, and rule processing terminates for that packet.
3. If the deny rule for the malicious subnet `198.51.100.0/24` were assigned a rule number higher than the general web allow rules (for example, rule 150 while TCP 80/443 were rules 100 and 110), an incoming HTTP packet from `198.51.100.9` destined for port 80 or 443 would match rule 100 or 110 first (`0.0.0.0/0` includes `198.51.100.0/24`). It would be granted entry before the NACL ever evaluated the deny rule, rendering the blacklist ineffective.
4. By assigning the deny rule a **lower rule number** (Rule 90) before the broader permissive rules (Rules 100 and 110), any packet originating from the abusive range `198.51.100.0/24` is evaluated first, matched immediately, and unconditionally dropped regardless of the destination port.
