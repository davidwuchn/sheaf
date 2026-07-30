### Generating Models with Macros

This example defines two networks from the same layer list:

```clojure
(defarchitectures
  [linear :h1 relu]
  [linear :h2 relu])
```

The first is a feed-forward network. The second adds a residual connection to
every layer. The macro also generates a training function for each model.

This is possible because Sheaf is homoiconic: the layer list is regular Sheaf
code, not a separate model description format. A macro can then transform it
into new Sheaf programs before compilation. The generated code is ordinary Sheaf
code as well.

In frameworks where model construction and compilation use separate APIs, this
step often happens through framework objects or graph capture. In Sheaf, model
generation is part of the language itself.
