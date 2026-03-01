// import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, AreaChart, Area, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Users, ShoppingCart, Package, Target, AlertTriangle, Lightbulb, ArrowUpRight, ArrowDownRight, Minus, Activity, Zap, RefreshCw, Download } from 'lucide-react';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import { useState, useEffect, useRef } from 'react';

// import"./index.css";

const API_BASE_URL = 'http://localhost:5010/api';

const ICON_MAP = {
  TrendingUp: TrendingUp,
  TrendingDown: TrendingDown,
  DollarSign: DollarSign,
  Users: Users,
  ShoppingCart: ShoppingCart,
  Package: Package,
  Target: Target,
  Activity: Activity
};

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

export default function KPIDashboard() {
  const [tableData, setTableData] = useState({ tables: [], views: [], grouped_tables: {}, table_groups: [] });
  const [selectedTable, setSelectedTable] = useState('');
  const [groupsData, setGroupsData] = useState([]); // Array to store dashboard data for all groups
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const dashboardRef = useRef(null);

  useEffect(() => {
    fetchTables();
  }, []);

  useEffect(() => {
    if (tableData.table_groups && tableData.table_groups.length > 0) {
      loadAllGroupsDashboards(tableData.table_groups);
    } else if (selectedTable) {
      loadSingleTableDashboard(selectedTable);
    }
  }, [tableData.table_groups, selectedTable]);

  const fetchTables = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/tables`);
      const data = await response.json();
      if (data.success) {
        const newTableData = { 
            tables: data.tables || [], 
            views: data.views || [], 
            grouped_tables: data.grouped_tables || {},
            table_groups: data.table_groups || []
        };
        setTableData(newTableData);
        // We no longer set selectedTable to a group immediately since we'll load all groups in the useEffect.
        // But if there are no groups, we default to the first table.
        if (!newTableData.table_groups || newTableData.table_groups.length === 0) {
          if (newTableData.tables && newTableData.tables.length > 0) {
            setSelectedTable(newTableData.tables[0]);
          } else if (newTableData.views && newTableData.views.length > 0) {
            setSelectedTable(newTableData.views[0]);
          }
        }
      }
    } catch (error) {
      console.error('Error fetching tables:', error);
    }
  };

  const loadAllGroupsDashboards = async (groups) => {
    setLoading(true);
    setGroupsData([]); // Clear existing
    
    try {
      const promises = groups.map(group => 
        fetch(`${API_BASE_URL}/dashboard/group`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tables: group.tables, group_name: group.group_name })
        }).then(res => res.json())
      );
      
      const results = await Promise.all(promises);
      const successfulGroups = results.filter(r => r.success);
      setGroupsData(successfulGroups.map(g => ({ kpiData: g, insights: null }))); // Start without insights
      setLastUpdated(new Date());

      // Optionally fetch insights for each group (in background)
      // For simplicity, we can skip individual group AI insights or load them incrementally
    } catch (error) {
      console.error('Error loading all group dashboards:', error);
    }
    setLoading(false);
  };

  const loadSingleTableDashboard = async (tableName) => {
    if (!tableName || tableName.startsWith('GROUP::')) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/kpi/${tableName}`);
      const data = await response.json();
      if (data.success) {
        setGroupsData([{ kpiData: data, insights: null }]);
        setLastUpdated(new Date());
        
        // Auto-generate insights
        generateInsights(0, tableName, data.kpi_values, data.chart_data);
      }
    } catch (error) {
      console.error('Error loading single KPI dashboard:', error);
    }
    setLoading(false);
  };

  const handleRefresh = () => {
    if (tableData.table_groups && tableData.table_groups.length > 0) {
      loadAllGroupsDashboards(tableData.table_groups);
    } else {
      loadSingleTableDashboard(selectedTable);
    }
  };

  const generateInsights = async (index, tableName, kpiValues, chartData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/kpi/insights/${tableName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kpi_values: kpiValues, chart_data: chartData })
      });
      const data = await response.json();
      if (data.success) {
        setGroupsData(prev => {
           const newArr = [...prev];
           newArr[index] = { ...newArr[index], insights: data.insights };
           return newArr;
        });
      }
    } catch (error) {
      console.error('Error generating insights:', error);
    }
  };

  const handleDownloadPDF = async () => {
    if (!dashboardRef.current) return;
    
    setDownloading(true);
    try {
      const canvas = await html2canvas(dashboardRef.current, {
        scale: 2,
        useCORS: true,
        logging: false,
        windowWidth: dashboardRef.current.scrollWidth,
        windowHeight: dashboardRef.current.scrollHeight,
        onclone: (clonedDoc) => {
          // Fix gradient text for PDF capture
          const title = clonedDoc.querySelector('h1');
          if (title) {
            title.style.background = 'none';
            title.style.webkitTextFillColor = 'initial'; 
            title.style.color = '#60a5fa'; // solid blue color
          }

          // Fix Select Dropdown display - replace with text
          const select = clonedDoc.querySelector('select');
          if (select) {
            const textSpan = clonedDoc.createElement('span');
            textSpan.innerText = select.value;
            textSpan.style.color = '#f8fafc'; // slate-50
            textSpan.style.fontSize = '1rem';
            textSpan.style.fontWeight = '500';
            textSpan.style.padding = '0.5rem 1rem';
            textSpan.style.backgroundColor = '#334155'; // slate-700
            textSpan.style.border = '1px solid #475569'; // slate-600
            textSpan.style.borderRadius = '0.5rem';
            select.parentNode.replaceChild(textSpan, select);
          }
        }
      });
      
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
      const imgX = (pdfWidth - imgWidth * ratio) / 2;
      const imgY = 10;
      
      // Calculate height based on width ratio to maintain aspect ratio
      const finalImgHeight = (imgHeight * pdfWidth) / imgWidth;
      
      // If content is longer than one page, we might need a different approach or multiple pages.
      // For now, let's fit to width and allow multi-page if needed (advanced) or just single page scaling.
      // Simple scaling to fit width:
      
      const imgHeightUpdated = (canvas.height * pdfWidth) / canvas.width;
      let heightLeft = imgHeightUpdated;
      let position = 0;

      pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, imgHeightUpdated);
      heightLeft -= pdfHeight;

      while (heightLeft >= 0) {
        position = heightLeft - imgHeightUpdated;
        pdf.addPage();
        pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, imgHeightUpdated);
        heightLeft -= pdfHeight;
      }
      
      pdf.save(`AI_Analytics_Report_${new Date().toISOString().split('T')[0]}.pdf`);
    } catch (error) {
      console.error('Error generating PDF:', error);
    }
    setDownloading(false);
  };

  const formatValue = (value, format) => {
    if (value === null || value === undefined) return 'N/A';
    
    switch (format) {
      case 'currency':
        return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      case 'percentage':
        return `${value.toFixed(2)}%`;
      case 'number':
        return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
      default:
        return value.toString();
    }
  };

  const getIcon = (iconName) => {
    const IconComponent = ICON_MAP[iconName] || Activity;
    return IconComponent;
  };

  const getTrendIcon = (direction) => {
    if (direction === 'up') return <ArrowUpRight className="text-green-400" size={20} />;
    if (direction === 'down') return <ArrowDownRight className="text-red-400" size={20} />;
    return <Minus className="text-slate-400" size={20} />;
  };

  const getCategoryColor = (category) => {
    const colors = {
      financial: 'from-green-500/10 to-emerald-600/10 border-green-500/20',
      operational: 'from-blue-500/10 to-blue-600/10 border-blue-500/20',
      customer: 'from-purple-500/10 to-purple-600/10 border-purple-500/20',
      product: 'from-orange-500/10 to-orange-600/10 border-orange-500/20'
    };
    return colors[category] || colors.operational;
  };

  const renderChart = (chartConfig) => {
    const { type, title, data, x_axis, y_axis } = chartConfig;

    if (!data || data.length === 0) return null;

    const chartProps = {
      data,
      margin: { top: 5, right: 30, left: 20, bottom: 5 }
    };

    switch (type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart {...chartProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey={x_axis} stroke="#9ca3af" angle={-45} textAnchor="end" height={80} />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Bar dataKey={y_axis} fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart {...chartProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey={x_axis} stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Line type="monotone" dataKey={y_axis} stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6' }} />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'area':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart {...chartProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey={x_axis} stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Area type="monotone" dataKey={y_axis} stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
            </AreaChart>
          </ResponsiveContainer>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data}
                dataKey={y_axis}
                nameKey={x_axis}
                cx="50%"
                cy="50%"
                outerRadius={100}
                label
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  const getPriorityBadge = (priority) => {
    const styles = {
      high: 'bg-red-500/20 text-red-400 border-red-500/30',
      medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      low: 'bg-green-500/20 text-green-400 border-green-500/30'
    };
    return styles[priority] || styles.medium;
  };

  if (loading && groupsData.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="animate-spin mx-auto mb-4 text-blue-400" size={48} />
          <p className="text-xl">Analyzing groups and generating KPIs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6" ref={dashboardRef}>
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
                AI-Powered KPI Dashboard
              </h1>
              <p className="text-slate-400">Intelligent metrics generated from your data</p>
            </div>
            {lastUpdated && (
              <div className="text-sm text-slate-400">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </div>
            )}
          </div>

          {/* Table Selector */}
          <div className="flex items-center gap-4 bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
            <label className="text-sm font-medium">Table:</label>
            <select
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <optgroup label="Tables (Individual)">
                {tableData.tables.map(table => (
                  <option key={table} value={table}>{table}</option>
                ))}
              </optgroup>
            </select>
            <div className="ml-auto flex gap-2" data-html2canvas-ignore="true">
              <button
                onClick={handleRefresh}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 px-4 py-2 rounded-lg flex items-center gap-2 transition-all"
              >
                <RefreshCw className={loading ? 'animate-spin' : ''} size={18} />
                Refresh
              </button>
              <button
                 onClick={handleDownloadPDF}
                 disabled={downloading || loading}
                 className="bg-purple-600 hover:bg-purple-700 disabled:opacity-50 px-4 py-2 rounded-lg flex items-center gap-2 transition-all text-white font-medium shadow-lg shadow-purple-900/20"
               >
                 <Download className={downloading ? 'animate-bounce' : ''} size={18} />
                 {downloading ? 'Downloading...' : 'Download Report'}
               </button>
            </div>
          </div>
        </div>

        <div className="space-y-16">
          {groupsData.length === 0 && !loading && (
             <div className="text-center py-20 text-slate-400">
               <p className="text-xl">No groups found or failed to generate dashboard.</p>
             </div>
          )}
          {groupsData.map((group, groupIdx) => {
             const { kpiData, insights } = group;
             if (!kpiData) return null;
             
             return (
               <div key={groupIdx} className="border border-slate-700/50 bg-slate-900/30 rounded-2xl p-6 shadow-2xl relative">
                  <div className="absolute top-0 right-0 bg-blue-600 text-xs px-3 py-1 rounded-bl-xl rounded-tr-xl font-medium shadow-lg">
                    {kpiData.table_name || `Dashboard ${groupIdx + 1}`}
                  </div>
                  
                  <h2 className="text-2xl font-semibold mb-6 pb-2 border-b border-slate-700 max-w-fit">
                    {kpiData.table_name || `Dashboard ${groupIdx + 1}`}
                  </h2>

                  <div className="space-y-6">
                    {/* KPI Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      {kpiData.kpi_values?.map((kpi, idx) => {
                        const IconComponent = getIcon(kpi.icon);
                        return (
                          <div
                            key={idx}
                            className={`bg-gradient-to-br ${getCategoryColor(kpi.category)} backdrop-blur-lg rounded-xl p-6 border transition-all hover:scale-105`}
                          >
                            <div className="flex items-start justify-between mb-3">
                              <div className="bg-slate-700/50 p-3 rounded-lg">
                                <IconComponent size={24} className="text-blue-400" />
                              </div>
                              {getTrendIcon(kpi.change_direction)}
                            </div>
                            <h3 className="text-sm font-medium text-slate-400 mb-1">{kpi.name}</h3>
                            <p className="text-3xl font-bold mb-1">{formatValue(kpi.value, kpi.format)}</p>
                            <p className="text-xs text-slate-500">{kpi.description}</p>
                          </div>
                        );
                      })}
                    </div>

                    {/* Charts */}
                    {kpiData.chart_data && kpiData.chart_data.length > 0 && (
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {kpiData.chart_data.map((chart, idx) => (
                          <div key={idx} className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50">
                            <div className="mb-4">
                              <h3 className="text-xl font-semibold mb-1">{chart.title}</h3>
                              <p className="text-sm text-slate-400">{chart.description}</p>
                            </div>
                            {renderChart(chart)}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* AI Insights */}
                    {insights && (
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Key Observations */}
                        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50">
                          <div className="flex items-center gap-2 mb-4">
                            <Activity className="text-blue-400" size={24} />
                            <h3 className="text-xl font-semibold">Key Observations</h3>
                          </div>
                          <div className="space-y-2">
                            {insights.observations?.map((obs, idx) => (
                              <div key={idx} className="flex items-start gap-3 bg-slate-700/30 rounded-lg p-3">
                                <div className="bg-blue-500/20 rounded-full p-1 mt-0.5">
                                  <div className="w-2 h-2 bg-blue-400 rounded-full" />
                                </div>
                                <p className="text-sm text-slate-300">{obs}</p>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Action Items */}
                        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50">
                          <div className="flex items-center gap-2 mb-4">
                            <Target className="text-purple-400" size={24} />
                            <h3 className="text-xl font-semibold">Action Items</h3>
                          </div>
                          <div className="space-y-3">
                            {insights.action_items?.map((item, idx) => (
                              <div key={idx} className="bg-slate-700/30 rounded-lg p-3">
                                <div className="flex items-center justify-between mb-1">
                                  <h4 className="font-medium text-sm">{item.title}</h4>
                                  <span className={`text-xs px-2 py-1 rounded-full border ${getPriorityBadge(item.priority)}`}>
                                    {item.priority}
                                  </span>
                                </div>
                                <p className="text-sm text-slate-400">{item.description}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
               </div>
             );
          })}
        </div>
      </div>
    </div>
  );
}