from trexdiff import Node, finite_diff

# More complex graph:
#
#  a --
#       mul1 = a * b --
#  b --               add2 = mul1 + mul2 --
#                                            mul3 = add2 * add1 => loss
#  c --               add1 = c + d ---------
#       mul2 = c * d --
#  d --

a = Node(1.5)
b = Node(2.0)
c = Node(0.5)
d = Node(3.0)

mul1 = a * b           # a * b
mul2 = c * d           # c * d
add1 = c + d           # c + d
add2 = mul1 + mul2     # (a*b) + (c*d)
loss = add2 * add1     # ((a*b)+(c*d)) * (c+d)

loss.forward()
loss.backward(1.0)

print("\n--- test ---")
print(f"loss: {loss.val:.6f}")
print(f"grad:  da={a.grad:.6f}, db={b.grad:.6f}, dc={c.grad:.6f}, dd={d.grad:.6f}")

print(f"fd:    da={finite_diff(a, loss):.6f}, db={finite_diff(b, loss):.6f}, "
      f"dc={finite_diff(c, loss):.6f}, dd={finite_diff(d, loss):.6f}")
