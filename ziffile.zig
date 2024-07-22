const std = @import("std");
const c = @cImport({
    @cInclude("some_ml_lib.h");
    @cInclude("another_ml_lib.h");
});

const DataType = enum {
    Float,
    Int,
    String,
};

const FeatureType = struct {
    name: []const u8,
    data_type: DataType,
};

const ModelType = enum {
    LinearRegression,
    RandomForest,
    NeuralNetwork,
};

const ModelConfig = struct {
    model_type: ModelType,
    hyperparameters: std.StringHashMap([]const u8),
};

const Dataset = struct {
    features: []FeatureType,
    data: [][]f64,
    labels: []f64,

    pub fn init(allocator: *std.mem.Allocator, features: []FeatureType, data: [][]f64, labels: []f64) !Dataset {
        return Dataset{
            .features = try allocator.dupe(FeatureType, features),
            .data = try allocator.dupe([]f64, data),
            .labels = try allocator.dupe(f64, labels),
        };
    }

    pub fn deinit(self: *Dataset, allocator: *std.mem.Allocator) void {
        allocator.free(self.features);
        for (self.data) |row| {
            allocator.free(row);
        }
        allocator.free(self.data);
        allocator.free(self.labels);
    }
};

const Model = struct {
    config: ModelConfig,
    trained: bool,
    internal_model: *c.SomeMLModel, // Placeholder for an external ML library's model representation

    pub fn init(config: ModelConfig) !Model {
        return Model{
            .config = config,
            .trained = false,
            .internal_model = try c.createModel(config.model_type),
        };
    }

    pub fn train(self: *Model, dataset: Dataset) !void {
        // Here we would interface with the external ML library to train the model
        // This is a placeholder implementation
        _ = c.trainModel(self.internal_model, dataset.data.ptr, dataset.labels.ptr, dataset.data.len);
        self.trained = true;
    }

    pub fn predict(self: *Model, input: []f64) !f64 {
        if (!self.trained) {
            return error.ModelNotTrained;
        }
        // Again, interfacing with external library
        return c.predict(self.internal_model, input.ptr, input.len);
    }

    pub fn deinit(self: *Model) void {
        c.destroyModel(self.internal_model);
    }
};

const CrossValidator = struct {
    n_folds: usize,
    shuffle: bool,

    pub fn init(n_folds: usize, shuffle: bool) CrossValidator {
        return CrossValidator{
            .n_folds = n_folds,
            .shuffle = shuffle,
        };
    }

    pub fn validate(self: CrossValidator, allocator: *std.mem.Allocator, model: *Model, dataset: Dataset) !f64 {
        var rng = std.rand.DefaultPrng.init(0);
        var indices = try allocator.alloc(usize, dataset.data.len);
        defer allocator.free(indices);

        for (indices) |*idx, i| {
            idx.* = i;
        }

        if (self.shuffle) {
            std.rand.Random.shuffle(rng.random(), usize, indices);
        }

        var total_score: f64 = 0;

        var fold: usize = 0;
        while (fold < self.n_folds) : (fold += 1) {
            var test_start = fold * dataset.data.len / self.n_folds;
            var test_end = if (fold == self.n_folds - 1) dataset.data.len else (fold + 1) * dataset.data.len / self.n_folds;

            var train_data = std.ArrayList([]f64).init(allocator);
            var train_labels = std.ArrayList(f64).init(allocator);
            var test_data = std.ArrayList([]f64).init(allocator);
            var test_labels = std.ArrayList(f64).init(allocator);

            defer train_data.deinit();
            defer train_labels.deinit();
            defer test_data.deinit();
            defer test_labels.deinit();

            for (indices) |idx, i| {
                if (i >= test_start and i < test_end) {
                    try test_data.append(dataset.data[idx]);
                    try test_labels.append(dataset.labels[idx]);
                } else {
                    try train_data.append(dataset.data[idx]);
                    try train_labels.append(dataset.labels[idx]);
                }
            }

            var train_dataset = try Dataset.init(allocator, dataset.features, train_data.items, train_labels.items);
            defer train_dataset.deinit(allocator);

            try model.train(train_dataset);

            var score: f64 = 0;
            for (test_data.items) |input, i| {
                var prediction = try model.predict(input);
                score += std.math.absFloat(prediction - test_labels.items[i]);
            }
            score /= @intToFloat(f64, test_data.items.len);

            total_score += score;
        }

        return total_score / @intToFloat(f64, self.n_folds);
    }
};

const FeatureSelector = struct {
    n_features: usize,

    pub fn init(n_features: usize) FeatureSelector {
        return FeatureSelector{
            .n_features = n_features,
        };
    }

    pub fn select(self: FeatureSelector, allocator: *std.mem.Allocator, dataset: Dataset) !Dataset {
        var scores = try allocator.alloc(f64, dataset.features.len);
        defer allocator.free(scores);

        // Calculate feature importance scores (placeholder implementation)
        for (dataset.features) |_, i| {
            scores[i] = c.calculateFeatureImportance(dataset.data.ptr, dataset.labels.ptr, dataset.data.len, i);
        }

        var selected_indices = try allocator.alloc(usize, self.n_features);
        defer allocator.free(selected_indices);

        // Select top N features
        var i: usize = 0;
        while (i < self.n_features) : (i += 1) {
            var max_score: f64 = -std.math.inf(f64);
            var max_index: usize = 0;
            for (scores) |score, j| {
                if (score > max_score) {
                    max_score = score;
                    max_index = j;
                }
            }
            selected_indices[i] = max_index;
            scores[max_index] = -std.math.inf(f64);
        }

        // Create new dataset with selected features
        var new_features = try allocator.alloc(FeatureType, self.n_features);
        var new_data = try allocator.alloc([]f64, dataset.data.len);

        for (selected_indices) |idx, j| {
            new_features[j] = dataset.features[idx];
            for (dataset.data) |row, k| {
                if (j == 0) {
                    new_data[k] = try allocator.alloc(f64, self.n_features);
                }
                new_data[k][j] = row[idx];
            }
        }

        return Dataset.init(allocator, new_features, new_data, dataset.labels);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = &gpa.allocator();

    // Example usage
    const features = [_]FeatureType{
        .{ .name = "feature1", .data_type = .Float },
        .{ .name = "feature2", .data_type = .Float },
        .{ .name = "feature3", .data_type = .Float },
    };

    const data = [_][]const f64{
        &[_]f64{ 1.0, 2.0, 3.0 },
        &[_]f64{ 4.0, 5.0, 6.0 },
        &[_]f64{ 7.0, 8.0, 9.0 },
    };

    const labels = [_]f64{ 0.0, 1.0, 1.0 };

    var dataset = try Dataset.init(allocator, &features, &data, &labels);
    defer dataset.deinit(allocator);

    var model_config = ModelConfig{
        .model_type = .RandomForest,
        .hyperparameters = std.StringHashMap([]const u8).init(allocator),
    };
    try model_config.hyperparameters.put("n_estimators", "100");
    try model_config.hyperparameters.put("max_depth", "10");

    var model = try Model.init(model_config);
    defer model.deinit();

    var cross_validator = CrossValidator.init(5, true);
    var score = try cross_validator.validate(allocator, &model, dataset);

    std.debug.print("Cross-validation score: {d}\n", .{score});

    var feature_selector = FeatureSelector.init(2);
    var selected_dataset = try feature_selector.select(allocator, dataset);
    defer selected_dataset.deinit(allocator);

    std.debug.print("Selected features: ", .{});
    for (selected_dataset.features) |feature| {
        std.debug.print("{s} ", .{feature.name});
    }
    std.debug.print("\n", .{});
}
