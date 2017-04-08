const path = require('path');
const webpack = require('webpack');

const SRC = path.resolve(__dirname, 'src');
const NODE_MODULES = path.resolve(__dirname, 'node_modules');

module.exports = {
	entry: path.join(SRC, 'js', 'index.js'),
	output: {
		path: path.join(__dirname, "public", "js"),
		filename: 'index.js'
	},
	watch: true,
	module: {
		rules: [{
			test: /\.jsx?$/,
			exclude: [NODE_MODULES],
			use: [{
				loader: 'babel-loader'
			}]
		}, {
			test: /\.json$/,
			use: [{
				loader: 'json-loader'
			}]
		}, {
			test: /\.svg$/,
			loaders: [{
				loader: 'babel-loader',
				query: {
					presets: ['es2015']
				}
			}, {
				loader: 'react-svg-loader',
				query: {
					jsx: true
				}
			}]
		}]
	},
	resolve: {
		modules: [
			path.join(SRC, 'js'),
			path.resolve(__dirname, 'data'),
			SRC,
			NODE_MODULES
		]
	}
};